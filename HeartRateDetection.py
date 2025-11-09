import tkinter as tk
from tkinter import messagebox
import threading
import numpy as np
import cv2
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.PlotModule import LivePlot
import cvzone
import time

# Define the dimensions of the webcam output and processing frame
realWidth = 640
realHeight = 480
videoWidth = 160
videoHeight = 120
videoChannels = 3
videoFrameRate = 15

# Parameters for Eulerian color magnification
levels = 3
alpha = 170
minFrequency = 1.0
maxFrequency = 2.0
bufferSize = 150
bufferIndex = 0

# BPM calculation configuration
bpmCalculationFrequency = 10
bpmBufferSize = 10
calculationDuration = 60

# Timer configuration
startTime = time.time()
stop_process = False

# FPS calculation
pTime = 0

# Initialize Gaussian pyramid for color magnification
videoGauss = None
fourierTransformAvg = None
frequencies = None
mask = None

def buildGauss(frame, levels):
    pyramid = [frame]
    for level in range(levels):
        frame = cv2.pyrDown(frame)
        pyramid.append(frame)
    return pyramid

def reconstructFrame(pyramid, index, levels):
    filteredFrame = pyramid[index]
    for level in range(levels):
        filteredFrame = cv2.pyrUp(filteredFrame)
    filteredFrame = filteredFrame[:videoHeight, :videoWidth]
    return filteredFrame

def process_frame(frame, detector):
    frame_float = frame.astype(float)
    b, g, r = cv2.split(frame_float)
    g = g * 1.5
    frame_processed = cv2.merge([b, g, r])
    frame_processed = np.clip(frame_processed, 0, 255)
    frame_processed = frame_processed.astype(np.uint8)
    return frame_processed

def start_monitor(avg_bpm_label, alert_label):
    global videoGauss, fourierTransformAvg, frequencies, mask, startTime, stop_process, pTime
    try:
        webcam = cv2.VideoCapture(0)
        detector = FaceDetector(minDetectionCon=0.7)
        plotY = LivePlot(realWidth, realHeight, [60, 120], invert=True)
        
        webcam.set(3, realWidth)
        webcam.set(4, realHeight)

        bpmBuffer = np.zeros((bpmBufferSize))
        bufferIndex = 0
        bpmBufferIndex = 0
        
        firstFrame = np.zeros((videoHeight, videoWidth, videoChannels))
        firstGauss = buildGauss(firstFrame, levels + 1)[levels]
        videoGauss = np.zeros((bufferSize, firstGauss.shape[0], firstGauss.shape[1], videoChannels))
        fourierTransformAvg = np.zeros((bufferSize))
        frequencies = (1.0 * videoFrameRate) * np.arange(bufferSize) / (1.0 * bufferSize)
        mask = (frequencies >= minFrequency) & (frequencies <= maxFrequency)

        while not stop_process:
            ret, frame = webcam.read()
            if not ret:
                break

            # Calculate FPS
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime

            # Detect face without drawing
            frame, bboxs = detector.findFaces(frame, draw=False)
            frameDraw = frame.copy()

            # Draw FPS
            cvzone.putTextRect(frameDraw, f'FPS: {int(fps)}', (20, 40), 
                             scale=2, thickness=2, colorR=(255,255,255))

            if bboxs:
                # Draw just the box without percentage
                bbox = bboxs[0]['bbox']
                x1, y1, w1, h1 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                cv2.rectangle(frameDraw, (x1, y1), (x1 + w1, y1 + h1), (255, 0, 255), 2)
                
                # Ensure coordinates are within frame bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                w1 = min(w1, frame.shape[1] - x1)
                h1 = min(h1, frame.shape[0] - y1)
                
                detectionFrame = frame[y1:y1 + h1, x1:x1 + w1]
                
                if detectionFrame.size != 0:
                    detectionFrame = cv2.resize(detectionFrame, (videoWidth, videoHeight))
                    processedFrame = process_frame(detectionFrame, detector)
                    
                    videoGauss[bufferIndex] = buildGauss(processedFrame, levels + 1)[levels]
                    fourierTransform = np.fft.fft(videoGauss, axis=0)
                    fourierTransform[mask == False] = 0

                    if bufferIndex % bpmCalculationFrequency == 0:
                        for buf in range(bufferSize):
                            fourierTransformAvg[buf] = np.real(fourierTransform[buf]).mean()
                        hz = frequencies[np.argmax(fourierTransformAvg)]
                        bpm = 60.0 * hz
                        bpmBuffer[bpmBufferIndex] = bpm
                        bpmBufferIndex = (bpmBufferIndex + 1) % bpmBufferSize

                    filtered = np.real(np.fft.ifft(fourierTransform, axis=0)) * alpha
                    filteredFrame = reconstructFrame(filtered, bufferIndex, levels)
                    outputFrame = detectionFrame + filteredFrame
                    outputFrame = cv2.convertScaleAbs(outputFrame)

                    bufferIndex = (bufferIndex + 1) % bufferSize

                    if bufferIndex > bpmBufferSize:
                        bpm_value = bpmBuffer.mean()
                        imgPlot = plotY.update(float(bpm_value))
                        cvzone.putTextRect(frameDraw, f'BPM: {bpm_value:.2f}', (300, 40), scale=2)
                        
                        app.after(0, lambda: avg_bpm_label.config(text=f"Average BPM: {bpm_value:.2f}"))
                        if bpm_value < 60:
                            app.after(0, lambda: alert_label.config(text="Alert: Low BPM!", fg="red"))
                            app.after(0, lambda: avg_bpm_label.config(fg="red"))
                        elif bpm_value > 100:
                            app.after(0, lambda: alert_label.config(text="Alert: High BPM!", fg="red"))
                            app.after(0, lambda: avg_bpm_label.config(fg="red"))
                        else:
                            app.after(0, lambda: alert_label.config(text="BPM Normal", fg="green"))
                            app.after(0, lambda: avg_bpm_label.config(fg="green"))
                    else:
                        imgPlot = plotY.update(0)
                        cvzone.putTextRect(frameDraw, "Calculating BPM...", (30, 80), scale=2)

                    # Create the layout with main frame, processed frame, and plot
                    processedFrame = cv2.resize(outputFrame, (160, 120))
                    h, w, _ = frameDraw.shape
                    overlay_h, overlay_w = processedFrame.shape[:2]
                    
                    # Position for processed frame (top-right corner)
                    x_offset = w - overlay_w - 10
                    y_offset = 10
                    
                    # Create ROI and overlay processed frame
                    roi = frameDraw[y_offset:y_offset+overlay_h, x_offset:x_offset+overlay_w]
                    frameDraw[y_offset:y_offset+overlay_h, x_offset:x_offset+overlay_w] = processedFrame

                    # Stack main frame with plot
                    imgStack = cvzone.stackImages([frameDraw, imgPlot], 2, 1)
                    cv2.imshow("Heart Rate Monitor", imgStack)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        webcam.release()
        cv2.destroyAllWindows()

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

def stop_monitor():
    global stop_process
    stop_process = True

def run_monitor(avg_bpm_label, alert_label):
    global stop_process, pTime
    stop_process = False
    pTime = time.time()
    thread = threading.Thread(target=start_monitor, args=(avg_bpm_label, alert_label))
    thread.daemon = True
    thread.start()

# Create main application window
app = tk.Tk()
app.title("Heart Rate Monitor")
app.geometry("500x350")
app.configure(bg="#f5f5f5")

# Create UI elements
label = tk.Label(app, text="Heart Rate Monitor", font=("Arial", 20, "bold"), bg="#f5f5f5", fg="#333")
label.pack(pady=20)

frame = tk.Frame(app, bg="#f5f5f5")
frame.pack(pady=10)

start_button = tk.Button(frame, text="Start Monitoring", font=("Arial", 14), bg="#4caf50", fg="white", 
                        padx=20, pady=10, relief="flat", 
                        command=lambda: run_monitor(avg_bpm_label, alert_label))
start_button.grid(row=0, column=0, padx=10)

stop_button = tk.Button(frame, text="Stop Monitoring", font=("Arial", 14), bg="#f44336", fg="white", 
                       padx=20, pady=10, relief="flat", command=stop_monitor)
stop_button.grid(row=0, column=1, padx=10)

avg_bpm_label = tk.Label(app, text="Average BPM: Not Calculated", font=("Arial", 14), bg="#f5f5f5", fg="#555")
avg_bpm_label.pack(pady=10)

alert_label = tk.Label(app, text="BPM Normal", font=("Arial", 12), bg="#f5f5f5", fg="green")
alert_label.pack(pady=10)

app.mainloop()