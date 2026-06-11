
try:
    import moviepy.editor as mp
    print("moviepy available")
except ImportError:
    print("moviepy missing")

try:
    import matplotlib.pyplot as plt
    print("matplotlib available")
except ImportError:
    print("matplotlib missing")

try:
    import librosa
    print("librosa available")
except ImportError:
    print("librosa missing")

try:
    import numpy as np
    print("numpy available")
except ImportError:
    print("numpy missing")


