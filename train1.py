import requests

url = 'http://127.0.0.1:5000/predict'
data = {
  "Age": 30,
  "Gender": "Male",
  "Income": 100000,
  "Education": "Bachelor's Degree",
  "Marital Status": "Single",
  "Number of Children": 2,
  "Home Ownership": "Rented"
}

response = requests.post(url, json=data)
print(response.json())
