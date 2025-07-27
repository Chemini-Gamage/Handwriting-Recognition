# Handwriting-Recognition
A deep learning-based handwriting recognition system that processes **single-line handwritten text images** . This project uses a CRNN model (CNN + BiLSTM + CTC loss) and a simple Streamlit interface.


Architecture: CRNN (CNN + BiLSTM + CTC loss)

Trained on: IAM Dataset

Output: Raw text transcription  

<img width="1150" height="795" alt="Screenshot 2025-07-27 205838" src="https://github.com/user-attachments/assets/f4fc9588-acae-4a5a-9ec3-23740395b977" />

<img width="603" height="284" alt="image" src="https://github.com/user-attachments/assets/4d60f6b1-dd10-44f1-aa50-028be8481f86" />

# How to start
# 1.Clone the repository   
git clone https://github.com/Chemini-Gamage/Handwriting-Recognition.git  
cd Handwriting-Recognition  
# 2.Activate a virtual environment   
python -m venv venv  
# Windows  
venv\Scripts\activate  
# macOS/Linux  
source venv/bin/activate  
# 3.Install the dependencies  
pip install -r requirements.txt  
# 4.How to run  
streamlit run app.py  
