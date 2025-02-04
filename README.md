# DoctorAI Backend

The **DoctorAI Backend** is a Flask-based application that provides two core functionalities:

- **/analyze** - Analyzes medical documents in various formats.  
- **/consult** - Allows users to consult with doctors based on their medical queries.

## 🚀 Features

- Analyze medical documents (PDF, DOCX, images).  
- Consult with doctors by specifying language, specialization, and medical questions.  
- Simple and scalable backend powered by Flask.

## 📋 Endpoints

### 1️⃣ `/analyze`

**Description:** Analyzes medical documents provided in different formats.

#### **Request Parameters:**

- `file` (required): The medical document to be analyzed. Supported formats:
  - `'jpg'`, `'jpeg'`, `'png'`, `'gif'`, `'docx'`, `'pdf'`
- `document_language` (required): Language of the document.  
- `answer_language` (required): Preferred language for the response.  
- `specialization` (optional): Specialization of the doctor needed for the analysis.

---

### 2️⃣ `/consult`

**Description:** Consult with a doctor by providing relevant details through a form.

#### **Form Parameters:**

- `language` (required): Preferred language for communication.  
- `specialization` (required): Specialization of the doctor for consultation.  
- `question` (required): Your medical question.

---

## ⚙️ Launching the Application

Ensure you have **Python** and **Flask** installed. Follow these steps to run the DoctorAI backend:

### Clone the Repository

```bash
git clone <repository-url>
cd <repository-directory>
```

### Create and Activate Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Flask Server

```bash
export FLASK_APP=app.py
export FLASK_ENV=development
flask run
```

**For Windows PowerShell:**

```bash
$env:FLASK_APP="app.py"
$env:FLASK_ENV="development"
flask run
```

### Access the Application

The backend server will be running at: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 📦 Dependencies

- **Flask** - Web framework for Python.  
- Additional packages listed in `requirements.txt`.

---

## 📝 Notes

- Ensure proper configurations for production environments.  
- Supported file formats: `'jpg'`, `'jpeg'`, `'png'`, `'gif'`, `'docx'`, `'pdf'`.

---

## 📧 Contact

For any issues, please raise an issue in the repository or contact the development team.

