# Rapid Prototype AI Project (Streamlit)

Rapid Prototype AI Project (Streamlit) เป็นโปรเจกต์สำหรับทดสอบและสร้างต้นแบบ (prototype) บน AI ผ่านเว็บแอปที่สร้างด้วย **Streamlit** ด้วย Python อย่างง่ายและรวดเร็ว

---

##  โครงสร้างไฟล์

- **app.py** – ไฟล์หลักของเว็บแอป Streamlit
- **requirements.txt** – รายชื่อ dependencies ที่โปรเจกต์ต้องใช้

---

##  คุณสมบัติหลัก (Features)

1. รันเว็บแอปที่พัฒนาโดยใช้ Python ผ่านคำสั่ง `streamlit`
2. รองรับการทดลองตัวอย่างหรือโมเดล AI ได้อย่างรวดเร็ว ไม่ต้องเขียนโค้ด frontend
3. มีโครงสร้างที่เรียบง่าย เหมาะสำหรับ prototype และทดสอบไอเดียเร็ว ๆ

---

##  Setup & Run


# 1. Clone Project
```bash
git clone https://github.com/17Saberz/Rapid_prototype_AI_Project-Streamlit-.git
```

# 2. Create virtual environment
```bash
python -m venv venv
```
# Windows:
```bash
venv\\Scripts\\activate
```
# macOS/Linux:
```bash
source venv/bin/activate
```

# 3. Install dependencies
```bash
pip install -r requirements.txt
```

# 4. Run on Streamlit
```bash
streamlit run app.py
```
