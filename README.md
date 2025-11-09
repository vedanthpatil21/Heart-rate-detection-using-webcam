# Heart-rate-detection-using-webcam
A computer vision project that calculates heart rate (BPM) from a webcam. Uses Python, OpenCV, cvzone, and NumPy to perform Eulerian Video Magnification (EVM) and FFT analysis on a detected face, tracking blood-flow pulses. Includes a live plot and a Tkinter GUI for controls.
# Real-Time Heart Rate Monitor using Webcam

This project detects a user's heart rate in real-time using a standard webcam. It uses computer vision to detect a face and applies Eulerian Video Magnification (EVM) to analyze subtle color changes in the skin, which correspond to blood flow.



## Features

* **Real-Time BPM:** Calculates Beats Per Minute (BPM) from the webcam feed.
* **Live Plot:** Displays a live graph of the heart rate over time.
* **Simple GUI:** An easy-to-use interface built with Tkinter to start and stop the monitor.
* **Health Alerts:** Shows "Low BPM," "High BPM," or "BPM Normal" based on the reading.

## How It Works

1.  **Face Detection:** Uses `cvzone`'s `FaceDetector` (based on MediaPipe) to find the user's face.
2.  **Region of Interest (ROI):** Selects the facial region for analysis.
3.  **Eulerian Magnification:** Applies a Gaussian pyramid and Fast Fourier Transform (FFT) to amplify the tiny color changes in the skin caused by blood pulsing.
4.  **Frequency Analysis:** Identifies the dominant frequency in the amplified signal within a valid range (1.0 to 2.0 Hz, corresponding to 60-120 BPM).
5.  **BPM Calculation:** Converts this frequency into Beats Per Minute (`BPM = frequency * 60`).
6.  **Display:** Overlays the BPM on the video feed and updates the Tkinter GUI.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
    cd YOUR_REPOSITORY_NAME
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the main Python script to launch the application:

```bash
python Grp182HeartRateDetection.py
