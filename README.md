# 🖐️ Real-Time Handwritten Digit Recognizer (OpenCV + PyTorch + MediaPipe)

This project is a **real-time digit recognition system** that lets you draw digits in the air using your **index finger**. It uses:

- **MediaPipe** for hand tracking
- **OpenCV** for capturing video and canvas drawing
- **PyTorch** to train and run a digit classification model (trained on MNIST)

---

## 📸 Demo

Draw a digit in the box using your index finger, then press:

- `w` — Toggle writing mode (start/stop drawing)
- `r` — Recognize the digit you drew
- `c` — Clear the canvas
- `Esc` — Exit the program

---

## 🔧 Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/licht005/real-time-digit-detector.git
   cd real-time-digit-detector
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Train the model (optional, already trained model is included):
   ```bash
   python train_model.py
   ```

4. Run the digit recognizer:
   ```bash
   python digit_recognizer.py
   ```

---

## 🧠 How It Works

- Hand tracking is done via **MediaPipe**, tracking the tip of the index finger.
- Drawing is done in a 280×280 region, which is resized to 28×28, centered, and preprocessed.
- A simple neural network trained on **MNIST** predicts the digit in real-time.
- Prediction confidence is displayed temporarily inside the drawing box.

---

## 🗂️ Files

- `digit_recognizer.py` — Main OpenCV + MediaPipe app
- `train_model.py` — PyTorch model training on MNIST
- `mnist_model.pth` — Trained model weights
- `requirements.txt` — Project dependencies
- `.gitignore` — Ignored files/folders
- `README.md` — Project overview and usage


---

## 📄 License

MIT License. Use, modify, and share freely.
