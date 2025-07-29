# 🧠 Brain Tumor Detection from MRI using Deep Learning

This project uses Convolutional Neural Networks (CNNs) to detect brain tumors from MRI scans. Built with **PyTorch**, **Streamlit**, and **Grad-CAM** for model explainability, the app allows users to upload MRI images and get predictions with visual insights into model decision-making.

![App Screenshot](sample_images/test1.jpg)

---

## 📂 Project Structure

brain-tumor-detection/
├── app.py # Streamlit frontend
├── utils.py # Model loading, preprocessing, prediction
├── gradcam.py # Grad-CAM visualizations
├── model/
│ └── brain_tumor_model.pth # Trained PyTorch model
├── data/
│ └── brain_tumor_dataset/ # Raw dataset with 'yes' and 'no' folders
├── scripts/
│ ├── train_model.py # CNN training script
│ └── prepare_data.py # Data preprocessing/augmentation
├── sample_images/
│ └── test1.jpg # Example test image
├── requirements.txt # Python dependencies
└── README.md # This file

---

## 🚀 Features

- ✅ Upload MRI image and get prediction (Tumor / No Tumor)
- ✅ Grad-CAM heatmaps for visual explanation
- ✅ Responsive UI with Streamlit
- ✅ Custom CNN architecture with transfer learning

---

## 🛠️ Installation & Setup

### 1. Clone the Repository

git clone https://github.com/hasinitarla/brain-tumor-detection.git
cd brain-tumor-detection

2. Create Virtual Environment (optional but recommended)

python -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate

3. Install Dependencies
4. pip install -r requirements.txt
   
🧪 Run the App

streamlit run app.py
Upload any MRI image from your local system or use sample_images/test1.jpg for testing.

🧠 Grad-CAM Explainability
This project uses Grad-CAM to generate attention heatmaps showing which parts of the MRI influenced the model's decision.

from gradcam import generate_gradcam
See gradcam.py for implementation.

📊 Model Training
Use the following to retrain or fine-tune the model:

python scripts/train_model.py
You can update the CNN architecture, optimizer, and learning rate in train_model.py.

📁 Dataset
The dataset should be organized like this:

brain_tumor_dataset/
├── yes/      # Images with tumor
└── no/       # Images without tumor
You can use any public dataset like Kaggle Brain MRI Images for Brain Tumor Detection.

☁️ Deployment
You can deploy this app using:
🔵 Azure App Services
🟣 Hugging Face Spaces (Gradio/Streamlit)
🟢 Streamlit Cloud (free tier)
Let me know if you want help deploying!

📜 License
This project is licensed under the MIT License — free to use and modify.

🙌 Acknowledgments
Dataset: Kaggle
PyTorch Team
Streamlit Community
---

### ✅ Tips:

- Save this file as `README.md` in your project root.
- You can also add badges, contact info, or GitHub Actions later.

