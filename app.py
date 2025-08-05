from flask import Flask, render_template, request, jsonify
import joblib
import os
from dotenv import load_dotenv
import requests

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Load model using relative path
model = joblib.load("model/rf_productivity_model.pkl")

# Define expected form fields in order
FEATURE_ORDER = [
    'efficiency',
    'idle_ratio',
    'overtime_ratio',
    'team_size',
    'avg_sentiment',
    'msg_count'
]

CLUSTER_MAP = {
    0: "Highly Efficient",
    1: "Moderate Performer",
    2: "Needs Improvement"
}

# In-memory employee data for leaderboard
employees = {}

# Standard times for tasks (in minutes for leaderboard)
standardTimes = {'Development': 120, 'Meetings': 60, 'Research': 180, 'Administration': 30, 'Other': 60}

def log_task(employee_id, task_data):
    if employee_id not in employees:
        employees[employee_id] = {'tasks': [], 'scores': {}}
    employees[employee_id]['tasks'].append(task_data)

def calculate_scores():
    for emp_id, data in employees.items():
        tasks = data['tasks']
        if not tasks:
            continue
        total_tasks = len(tasks)
        avg_time = sum(t['time_spent'] for t in tasks) / total_tasks
        completion_rate = sum(1 for t in tasks if t['time_spent'] * 60 <= standardTimes.get(t['task_category'], 120)) / total_tasks
        
        # Weighted score
        data['scores']['total'] = (0.4 * total_tasks) + (0.3 * (1 / avg_time if avg_time > 0 else 0)) + (0.3 * completion_rate)

def get_leaderboard():
    calculate_scores()
    sorted_employees = sorted(employees.items(), key=lambda x: x[1]['scores'].get('total', 0), reverse=True)
    return sorted_employees[:5]  # Top 5

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Validate and extract input data
        input_data = []
        for field in FEATURE_ORDER:
            value = request.form.get(field)
            if not value:
                return render_template('index.html', 
                                     result=f"Missing value for {field.replace('_', ' ')}")
            try:
                input_data.append(float(value))
            except ValueError:
                return render_template('index.html',
                                      result=f"Invalid number for {field.replace('_', ' ')}")

        # Make prediction
        prediction = model.predict([input_data])[0]
        result = CLUSTER_MAP.get(prediction, f"Unknown Cluster ({prediction})")

        return render_template('index.html', result=result)

    except Exception as e:
        # Consider logging the error here
        return render_template('index.html', 
                             result="An error occurred during prediction. Please check your inputs.")

@app.route('/log_task', methods=['POST'])
def log_task_route():
    data = request.json
    employee_id = data.get('employee_id', 'default')  # Add employee_id to your task form
    task_data = {
        'task_category': data['task_category'],
        'time_spent': float(data['time_spent']),
        'wellness': data.get('wellness', 'Neutral'),
        'sentiment': data.get('sentiment', 'Neutral')
    }
    log_task(employee_id, task_data)
    return {'status': 'success'}

@app.route('/leaderboard')
def leaderboard():
    top_employees = get_leaderboard()
    return render_template('leaderboard.html', top_employees=top_employees)

@app.route('/chat', methods=['POST'])
def chat():
    message = request.json['message']
    api_key = os.getenv('OPENROUTER_API_KEY')
    try:
        response = requests.post(
            'https://openrouter.ai/api/v1/chat/completions',
            headers={
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            },
            json={
                'model': 'openai/gpt-3.5-turbo',
                'messages': [{'role': 'user', 'content': message}]
            }
        )
        response.raise_for_status()
        bot_reply = response.json()['choices'][0]['message']['content']
        return {'reply': bot_reply}
    except Exception as e:
        return {'error': 'Could not connect to AI service.'}, 500

if __name__ == '__main__':
    app.run(debug=True)  # Remove debug=True in production