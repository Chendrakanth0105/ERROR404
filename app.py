from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__,static_folder='static')

G2 = pickle.load(open('model_G2.pkl', 'rb'))
G3 = pickle.load(open('model_G3.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about',methods=['GET'])
def about():
    return render_template('aboutus.html')
    
@app.route('/predict', methods=['POST', 'GET'])
def predict():
    try:
        age = int(request.form.get('Age'))
        Medu = int(request.form.get('Medu'))
        Fedu = int(request.form.get('Fedu'))
        traveltime = int(request.form.get('traveltime'))
        studytime = int(request.form.get('studytime'))
        failures = int(request.form.get('failures'))
        goout = int(request.form.get('goout'))
        Dalc = int(request.form.get('Dalc'))
        Walc = int(request.form.get('Walc'))
        absences = int(request.form.get('absences'))
        school_LVA = (request.form.get('school_LVA'))
        school_MS = (request.form.get('school_MS'))
        G1 = int(request.form.get('G1'))
        address_U = (request.form.get('address_U'))
        Mjob_health = (request.form.get('Mjob_health'))
        Mjob_teacher = (request.form.get('Mjob_teacher'))
        Fjob_teacher = (request.form.get('Fjob_teacher'))
        reason_reputation = (request.form.get('reason_reputation'))
        schoolsup_yes = (request.form.get('schoolsup_yes'))
        higher_yes = (request.form.get('higher_yes'))
        internet_yes = (request.form.get('internet_yes'))
        result = G3.predict([[age, Medu, Fedu, traveltime, studytime, failures, goout, Dalc, Walc, absences, G1, school_LVA, school_MS, address_U, Mjob_health, Mjob_teacher, Fjob_teacher, reason_reputation, schoolsup_yes, higher_yes, internet_yes]])

        return render_template('result.html', prediction=result)
    except ValueError:
        # Handle the case where the input is not a valid integer
        return render_template('error.html', message="Invalid input for age. Please enter a valid number.")

    except Exception as e:
        # Handle other exceptions
        return render_template('error.html', message=str(e))
    # if result == 1:
    #      return render_template('index.html',label = 1)
    # else:
    #      return render_template('index.html',label = -1)

if __name__ == '__main__':
    app.run(debug=True)
