import sys
import requests
import json
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, 
    QLineEdit, QLabel, QCheckBox, QComboBox, QMessageBox, QStackedWidget, 
    QListWidget, QFileDialog, QTabWidget, QFormLayout, QHBoxLayout
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

API_URL = "http://127.0.0.1:8000"  # FastAPI address

# Custom styles for the application
STYLESHEET = """
    QWidget {
        font-family: "Segoe UI";
        font-size: 14px;
    }
    QPushButton {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px;
        border-radius: 5px;
    }
    QPushButton:hover {
        background-color: #45a049;
    }
    QLineEdit, QComboBox, QCheckBox {
        padding: 5px;
        border: 1px solid #ccc;
        border-radius: 5px;
    }
    QListWidget {
        border: 1px solid #ccc;
        border-radius: 5px;
        padding: 5px;
    }
    QLabel {
        font-weight: bold;
    }
"""

class LoginWindow(QWidget):
    def __init__(self, switch_callback):
        super().__init__()
        self.switch_callback = switch_callback
        layout = QVBoxLayout()

        # Title
        title = QLabel("Cardiovascular Disease Prediction")
        title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Email input
        layout.addWidget(QLabel("Email"))
        self.email_input = QLineEdit()
        self.email_input.setPlaceholderText("Enter your email")
        layout.addWidget(self.email_input)

        # Password input
        layout.addWidget(QLabel("Password"))
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("Enter your password")
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        layout.addWidget(self.password_input)

        # Login button
        self.login_button = QPushButton("Login")
        self.login_button.clicked.connect(self.login)
        layout.addWidget(self.login_button)

        # Register button
        self.register_button = QPushButton("Register")
        self.register_button.clicked.connect(self.register)
        layout.addWidget(self.register_button)

        self.setLayout(layout)

    def login(self):
        email = self.email_input.text()
        password = self.password_input.text()
        try:
            response = requests.post(f"{API_URL}/login", json={"email": email, "password": password})
            if response.status_code == 200:
                self.switch_callback("main", {"email": email})
            else:
                QMessageBox.warning(self, "Error", "Invalid credentials")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to connect: {e}")

    def register(self):
        email = self.email_input.text()
        password = self.password_input.text()
        try:
            response = requests.post(f"{API_URL}/register", json={"email": email, "password": password})
            if response.status_code == 200:
                QMessageBox.information(self, "Success", "User registered successfully")
            else:
                QMessageBox.warning(self, "Error", "User already exists")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to connect: {e}")

class MainWindow(QMainWindow):
    def __init__(self, user_data, switch_callback):
        super().__init__()
        self.switch_callback = switch_callback
        self.user_data = user_data

        self.setWindowTitle("Cardiovascular Disease Prediction")
        self.setGeometry(100, 100, 800, 600)

        # Main layout
        main_layout = QVBoxLayout()

        # Title
        title = QLabel(f"Welcome, {self.user_data['email']}")
        title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title)

        # Tab widget for Manual Input and JSON Upload
        self.tabs = QTabWidget()
        self.tabs.addTab(self.create_manual_input_tab(), "Manual Input")
        self.tabs.addTab(self.create_json_upload_tab(), "Upload JSON")
        main_layout.addWidget(self.tabs)

        # Result display
        self.result_label = QLabel("")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.result_label)

        # Profile button
        profile_button = QPushButton("Profile")
        profile_button.clicked.connect(lambda: self.switch_callback("profile", self.user_data))
        main_layout.addWidget(profile_button)

        # Logout button
        logout_button = QPushButton("Logout")
        logout_button.clicked.connect(self.logout)
        main_layout.addWidget(logout_button)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def create_manual_input_tab(self):
        tab = QWidget()
        layout = QFormLayout()

        # Input fields
        self.inputs = {}
        fields = [
            ("Age", "age"), ("Height (cm)", "height"), ("Weight (kg)", "weight"),
            ("Systolic BP", "ap_hi"), ("Diastolic BP", "ap_lo")
        ]
        for label, key in fields:
            self.inputs[key] = QLineEdit()
            layout.addRow(label, self.inputs[key])

        # Dropdowns
        self.gender_box = QComboBox()
        self.gender_box.addItems(["Male", "Female"])
        layout.addRow("Gender", self.gender_box)

        self.cholesterol_box = QComboBox()
        self.cholesterol_box.addItems(["Normal", "Above Normal", "Well Above Normal"])
        layout.addRow("Cholesterol", self.cholesterol_box)

        self.glucose_box = QComboBox()
        self.glucose_box.addItems(["Normal", "Above Normal", "Well Above Normal"])
        layout.addRow("Glucose", self.glucose_box)

        self.activity_box = QComboBox()
        self.activity_box.addItems(["Low", "Moderate", "High"])
        layout.addRow("Activity Level", self.activity_box)

        # Checkboxes
        self.smoke_checkbox = QCheckBox("Smoker")
        self.alcohol_checkbox = QCheckBox("Alcohol Consumer")
        layout.addRow(self.smoke_checkbox)
        layout.addRow(self.alcohol_checkbox)

        # Analyze button
        analyze_button = QPushButton("Analyze")
        analyze_button.clicked.connect(self.analyze)
        layout.addRow(analyze_button)

        tab.setLayout(layout)
        return tab

    def create_json_upload_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        # Upload JSON button
        upload_button = QPushButton("Upload JSON")
        upload_button.clicked.connect(self.upload_json)
        layout.addWidget(upload_button)

        tab.setLayout(layout)
        return tab

    def analyze(self):
        try:
            input_data = [
                float(self.inputs["age"].text()),
                1 if self.gender_box.currentText() == "Male" else 2,
                float(self.inputs["height"].text()),
                float(self.inputs["weight"].text()),
                float(self.inputs["ap_hi"].text()),
                float(self.inputs["ap_lo"].text()),
                self.cholesterol_box.currentIndex() + 1,
                self.glucose_box.currentIndex() + 1,
                int(self.smoke_checkbox.isChecked()),
                int(self.alcohol_checkbox.isChecked()),
                self.activity_box.currentIndex() + 1
            ]
            response = requests.post(f"{API_URL}/analyze", json={"email": self.user_data["email"], "data": input_data})
            if response.status_code == 200:
                result = response.json()["result"]
                self.result_label.setText(f"Result: {'High Risk' if result else 'Low Risk'}")
            else:
                QMessageBox.warning(self, "Error", "Analysis failed")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Invalid input: {e}")

    def upload_json(self):
        try:
            file_path, _ = QFileDialog.getOpenFileName(self, "Open JSON File", "", "JSON Files (*.json)")
            if file_path:
                with open(file_path, "r") as file:
                    data = json.load(file)
                response = requests.post(f"{API_URL}/analyze/json", files={"file": open(file_path, "rb")}, data={"email": self.user_data["email"]})
                if response.status_code == 200:
                    result = response.json()["result"]
                    self.result_label.setText(f"Result: {'High Risk' if result else 'Low Risk'}")
                else:
                    QMessageBox.warning(self, "Error", "Analysis failed")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Invalid JSON file: {e}")

    def logout(self):
        self.switch_callback("login")

class ProfileWindow(QWidget):
    def __init__(self, user_data, switch_callback):
        super().__init__()
        self.switch_callback = switch_callback
        self.user_data = user_data

        layout = QVBoxLayout()

        # Title
        title = QLabel("Profile")
        title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # History
        self.history_list = QListWidget()
        layout.addWidget(QLabel("History"))
        layout.addWidget(self.history_list)
        self.load_history()

        # Back button
        back_button = QPushButton("Back to Main")
        back_button.clicked.connect(lambda: self.switch_callback("main", self.user_data))
        layout.addWidget(back_button)

        self.setLayout(layout)

    def load_history(self):
        try:
            response = requests.get(f"{API_URL}/history/{self.user_data['email']}")
            if response.status_code == 200:
                self.history_list.clear()
                for analysis in response.json()["history"]:
                    self.history_list.addItem(f"Result: {analysis['result']}, Date: {analysis['timestamp']}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load history: {e}")

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cardiovascular Disease Prediction")
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet(STYLESHEET)

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.login_window = LoginWindow(self.switch_screen)
        self.stack.addWidget(self.login_window)

    def switch_screen(self, screen_name, user_data=None):
        if screen_name == "main":
            self.main_window = MainWindow(user_data, self.switch_screen)
            self.stack.addWidget(self.main_window)
            self.stack.setCurrentWidget(self.main_window)
        elif screen_name == "profile":
            self.profile_window = ProfileWindow(user_data, self.switch_screen)
            self.stack.addWidget(self.profile_window)
            self.stack.setCurrentWidget(self.profile_window)
        elif screen_name == "login":
            self.stack.setCurrentWidget(self.login_window)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec())