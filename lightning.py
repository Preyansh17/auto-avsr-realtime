import torch
import torchaudio
from cosine import WarmupCosineScheduler
from datamodule.transforms import TextTransform

from pytorch_lightning import LightningModule
from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.pytorch_backend.e2e_asr_conformer import E2E
from espnet.nets.scorers.length_bonus import LengthBonus
from espnet.nets.scorers.ctc import CTCPrefixScorer
from modules.lora import inject_lora


def compute_word_level_distance(seq1, seq2):
    return torchaudio.functional.edit_distance(seq1.lower().split(), seq2.lower().split())


class ModelModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        if self.cfg.data.modality == "audio":
            self.backbone_args = self.cfg.model.audio_backbone
        elif self.cfg.data.modality == "video":
            self.backbone_args = self.cfg.model.visual_backbone

        self.text_transform = TextTransform()
        self.token_list = self.text_transform.token_list
        self.model = E2E(len(self.token_list), self.backbone_args)

        # Load vocabulary constraint if specified
        self.allowed_token_ids = None
        if hasattr(self.cfg, 'vocab_file') and self.cfg.vocab_file:
            self.allowed_token_ids = self._load_vocab_constraint(self.cfg.vocab_file)
            print(f"\n✓ Loaded vocabulary constraint: {len(self.allowed_token_ids)} allowed tokens")

        # -- initialise
        if self.cfg.pretrained_model_path:
            ckpt = torch.load(self.cfg.pretrained_model_path, map_location=lambda storage, loc: storage)
            if self.cfg.transfer_frontend:
                tmp_ckpt = {k: v for k, v in ckpt["model_state_dict"].items() if k.startswith("trunk.") or k.startswith("frontend3D.")}
                self.model.encoder.frontend.load_state_dict(tmp_ckpt)
            elif self.cfg.transfer_encoder:
                tmp_ckpt = {k.replace("encoder.", ""): v for k, v in ckpt.items() if k.startswith("encoder.")}
                self.model.encoder.load_state_dict(tmp_ckpt, strict=True)
            else:
                self.model.load_state_dict(ckpt)

        # Optionally inject LoRA and freeze base weights
        self.lora_enabled = getattr(self.cfg, "lora", {}).get("enabled", False) if hasattr(self.cfg, "lora") else False
        if self.lora_enabled:
            r = getattr(self.cfg.lora, "r", 8)
            alpha = getattr(self.cfg.lora, "alpha", 16)
            dropout = getattr(self.cfg.lora, "dropout", 0.0)
            scopes = getattr(self.cfg.lora, "scopes", ["encoder", "decoder"])
            patterns = getattr(self.cfg.lora, "name_patterns", [])
            pattern_scopes = getattr(self.cfg.lora, "pattern_scopes", None)
            replaced = inject_lora(self.model, scopes, r=r, alpha=alpha, dropout=dropout, 
                                 name_patterns=patterns, pattern_scopes=pattern_scopes)
            # Freeze base parameters; train only LoRA
            for n, p in self.model.named_parameters():
                if "lora_A" in n or "lora_B" in n:
                    p.requires_grad = True
                else:
                    p.requires_grad = False
            print(f"\n✓ LoRA enabled (video): replaced {replaced} Linear layers; training only LoRA params")
        
        # Optionally freeze specific layers for full finetuning (no LoRA)
        self.freeze_enabled = getattr(self.cfg, "freeze", {}).get("enabled", False) if hasattr(self.cfg, "freeze") else False
        if self.freeze_enabled and not self.lora_enabled:
            freeze_scopes = getattr(self.cfg.freeze, "scopes", [])
            freeze_patterns = getattr(self.cfg.freeze, "name_patterns", [])
            freeze_pattern_scopes = getattr(self.cfg.freeze, "pattern_scopes", None)
            frozen_count = self._freeze_parameters(freeze_scopes, freeze_patterns, freeze_pattern_scopes)
            print(f"\n✓ Freeze enabled (video): froze {frozen_count} parameters; training remaining params")
    
    def _freeze_parameters(self, scopes, name_patterns, pattern_scopes):
        """Freeze parameters matching the specified scopes and patterns."""
        import re
        frozen_count = 0
        
        for name, param in self.model.named_parameters():
            should_freeze = False
            
            # Check if name matches any scope
            for scope in scopes:
                if name.startswith(f"{scope}."):
                    should_freeze = True
                    break
            
            # Check if name matches any pattern
            if name_patterns:
                for pattern in name_patterns:
                    # If pattern_scopes is specified, only apply pattern to those scopes
                    if pattern_scopes:
                        for scope in pattern_scopes:
                            if name.startswith(f"{scope}.") and re.search(pattern, name):
                                should_freeze = True
                                break
                    else:
                        # Apply pattern to all names
                        if re.search(pattern, name):
                            should_freeze = True
                            break
            
            if should_freeze:
                param.requires_grad = False
                frozen_count += 1
        
        return frozen_count
    
    def _load_vocab_constraint(self, vocab_file):
        """Load vocabulary constraint from file and return allowed token IDs."""
        allowed_tokens = set()
        with open(vocab_file, 'r') as f:
            for line in f:
                sentence = line.strip()
                if sentence:
                    # 重要：处理句子格式 - 转大写并替换空格为▁
                    processed_sentence = sentence.upper().replace(" ", "▁")
                    token_ids = self.text_transform.tokenize(processed_sentence)
                    token_ids_list = token_ids if isinstance(token_ids, list) else token_ids.tolist()
                    allowed_tokens.update(token_ids_list)
        # 添加 blank token (0) - CTC需要
        allowed_tokens.add(0)
        return sorted(list(allowed_tokens))

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW([
            {"name": "lora" if self.lora_enabled else "model", "params": params, "lr": self.cfg.optimizer.lr}
        ], weight_decay=self.cfg.optimizer.weight_decay, betas=(0.9, 0.98))
        scheduler = WarmupCosineScheduler(optimizer, self.cfg.optimizer.warmup_epochs, self.cfg.trainer.max_epochs, len(self.trainer.datamodule.train_dataloader()))
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    def forward(self, sample, allowed_token_ids=None, return_tokens=False):
        # Get beam search parameters from config if available
        beam_size = getattr(self.cfg, 'beam_size', 40)
        pre_beam_ratio = getattr(self.cfg, 'pre_beam_ratio', 1.5)
        ctc_weight = getattr(self.cfg, 'ctc_weight', 0.1)
        
        self.beam_search = get_beam_search_decoder(
            self.model, 
            self.token_list, 
            ctc_weight=ctc_weight,
            beam_size=beam_size,
            pre_beam_ratio=pre_beam_ratio,
            allowed_token_ids=allowed_token_ids
        )
        enc_feat, _ = self.model.encoder(sample.unsqueeze(0).to(self.device), None)
        enc_feat = enc_feat.squeeze(0)

        nbest_hyps = self.beam_search(enc_feat)
        nbest_hyps = [h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]]
        predicted_token_id = torch.tensor(list(map(int, nbest_hyps[0]["yseq"][1:])))
        predicted = self.text_transform.post_process(predicted_token_id).replace("<eos>", "")
        
        if return_tokens:
            return predicted, predicted_token_id.tolist()
        return predicted

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, step_type="train")

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx, step_type="val")
        
        # Decode predictions for WER calculation and logging for all validation samples
        try:
            dataset_ids = batch.get("dataset_ids", None)
            if dataset_ids is not None:
                dataset_ids = dataset_ids.view(-1).tolist()
            else:
                dataset_ids = [0] * len(batch["inputs"])

            # Decode each sample in batch using vocab constraint if available
            for i in range(len(batch["inputs"])):
                sample_input = batch["inputs"][i:i+1]
                sample_target = batch["targets"][i]
                dataset_id = int(dataset_ids[i]) if i < len(dataset_ids) else 0

                # Get prediction (uses self.allowed_token_ids if available)
                predicted = self.forward(sample_input.squeeze(0), allowed_token_ids=self.allowed_token_ids)

                # Get ground truth
                ground_truth = self.text_transform.post_process(sample_target)

                # Calculate WER for this sample
                distance = compute_word_level_distance(ground_truth, predicted)
                length = len(ground_truth.split())

                self.val_total_edit_distance += distance
                self.val_total_length += length

                source_name = "other"
                if dataset_id == 1:
                    source_name = "legal298"
                    self.val_total_edit_distance_legal298 += distance
                    self.val_total_length_legal298 += length
                elif dataset_id == 2:
                    source_name = "legacy"
                    self.val_total_edit_distance_legacy += distance
                    self.val_total_length_legacy += length

                # Store for table logging
                self.val_predictions.append(predicted)
                self.val_ground_truths.append(ground_truth)
                self.val_sources.append(source_name)
        except Exception:
            # If decoding fails, skip sample(s) to keep training running
            pass
        
        return loss

    def test_step(self, sample, sample_idx):
        enc_feat, _ = self.model.encoder(sample["input"].unsqueeze(0).to(self.device), None)
        enc_feat = enc_feat.squeeze(0)

        nbest_hyps = self.beam_search(enc_feat)
        nbest_hyps = [h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]]
        predicted_token_id = torch.tensor(list(map(int, nbest_hyps[0]["yseq"][1:])))
        predicted = self.text_transform.post_process(predicted_token_id).replace("<eos>", "")

        token_id = sample["target"]
        actual = self.text_transform.post_process(token_id)

        self.total_edit_distance += compute_word_level_distance(actual, predicted)
        self.total_length += len(actual.split())
        return

    def _step(self, batch, batch_idx, step_type):
        loss, loss_ctc, loss_att, acc = self.model(batch["inputs"], batch["input_lengths"], batch["targets"])
        batch_size = len(batch["inputs"])

        if step_type == "train":
            self.log("loss", loss, on_step=True, on_epoch=True, batch_size=batch_size)
            self.log("loss_ctc", loss_ctc, on_step=False, on_epoch=True, batch_size=batch_size)
            self.log("loss_att", loss_att, on_step=False, on_epoch=True, batch_size=batch_size)
            self.log("decoder_acc", acc, on_step=True, on_epoch=True, batch_size=batch_size)
        else:
            self.log("loss_val", loss, batch_size=batch_size)
            self.log("loss_ctc_val", loss_ctc, batch_size=batch_size)
            self.log("loss_att_val", loss_att, batch_size=batch_size)
            self.log("decoder_acc_val", acc, batch_size=batch_size)

        if step_type == "train":
            self.log("monitoring_step", torch.tensor(self.global_step, dtype=torch.float32))

        return loss

    def on_train_epoch_start(self):
        sampler = self.trainer.train_dataloader.loaders.batch_sampler
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(self.current_epoch)
        return super().on_train_epoch_start()

    def on_validation_epoch_start(self):
        self.val_predictions = []
        self.val_ground_truths = []
        self.val_sources = []
        self.val_total_length = 0
        self.val_total_edit_distance = 0
        self.val_total_length_legal298 = 0
        self.val_total_edit_distance_legal298 = 0
        self.val_total_length_legacy = 0
        self.val_total_edit_distance_legacy = 0
        
    def on_validation_epoch_end(self):
        if self.val_total_length > 0:
            val_wer = self.val_total_edit_distance / self.val_total_length
            self.log("val_wer", val_wer, prog_bar=True)

            # Split reporting only appears for merged dataset samples (dataset_id 1/2).
            if self.val_total_length_legal298 > 0:
                val_wer_legal298 = (
                    self.val_total_edit_distance_legal298 / self.val_total_length_legal298
                )
                self.log("val_wer_legal298", val_wer_legal298, prog_bar=True)
            if self.val_total_length_legacy > 0:
                val_wer_legacy = (
                    self.val_total_edit_distance_legacy / self.val_total_length_legacy
                )
                self.log("val_wer_legacy", val_wer_legacy, prog_bar=True)
            
            # Log predictions table to wandb
            if len(self.val_predictions) > 0 and hasattr(self.logger, 'experiment'):
                import wandb
                # Create table with all predictions and ground truths
                table_data = []
                for i, (pred, gt, src) in enumerate(
                    zip(self.val_predictions, self.val_ground_truths, self.val_sources)
                ):
                    # Calculate WER for this sample
                    import torchaudio
                    sample_wer = torchaudio.functional.edit_distance(
                        gt.lower().split(), pred.lower().split()
                    ) / max(len(gt.split()), 1)
                    table_data.append([i, src, gt, pred, sample_wer])
                
                table = wandb.Table(
                    columns=["sample_id", "source", "ground_truth", "prediction", "wer"],
                    data=table_data
                )
                self.logger.experiment.log({
                    f"validation_predictions_epoch_{self.current_epoch}": table
                })
    
    def on_test_epoch_start(self):
        self.total_length = 0
        self.total_edit_distance = 0
        self.text_transform = TextTransform()
        self.beam_search = get_beam_search_decoder(self.model, self.token_list)

    def on_test_epoch_end(self):
        self.log("wer", self.total_edit_distance / self.total_length)


def get_beam_search_decoder(model, token_list, ctc_weight=0.1, beam_size=40, pre_beam_ratio=1.5, allowed_token_ids=None):
    """
    Get beam search decoder with optional vocabulary constraint.
    
    Args:
        model: The ASR model
        token_list: List of all tokens
        ctc_weight: Weight for CTC score (default: 0.1)
        beam_size: Beam size for search (default: 40)
        pre_beam_ratio: Pre-beam size ratio (default: 1.5, i.e., pre_beam_size = int(1.5 * beam_size) = 60)
        allowed_token_ids: Optional list of allowed token IDs for constrained decoding.
                          If None, all tokens are allowed (default behavior).
    
    Returns:
        BatchBeamSearch decoder
    """
    scorers = {
        "decoder": model.decoder,
        "ctc": CTCPrefixScorer(model.ctc, model.eos),
        "length_bonus": LengthBonus(len(token_list)),
        "lm": None
    }

    weights = {
        "decoder": 1.0 - ctc_weight,
        "ctc": ctc_weight,
        "lm": 0.0,
        "length_bonus": 0.0,
    }

    return BatchBeamSearch(
        beam_size=beam_size,
        vocab_size=len(token_list),
        weights=weights,
        scorers=scorers,
        sos=model.sos,
        eos=model.eos,
        token_list=token_list,
        pre_beam_ratio=pre_beam_ratio,
        pre_beam_score_key=None if ctc_weight == 1.0 else "decoder",
        allowed_token_ids=allowed_token_ids,
    )
