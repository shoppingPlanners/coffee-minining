def generate_health_advice(user_row: dict):
    """
    Generate simple advice based on user features.
    user_row: dict with keys like 'BMI', 'Sleep_Hours', 'Stress_Level', 'Coffee_Intake'
    """
    advice = []
    
    if user_row.get('BMI', 0) > 25:
        advice.append("Consider a healthier diet and regular exercise.")
    if user_row.get('Sleep_Hours', 8) < 7:
        advice.append("Try to get at least 7-8 hours of sleep.")
    if user_row.get('Stress_Level', 0) > 7:
        advice.append("Practice stress management techniques.")
    if user_row.get('Coffee_Intake', 0) > 3:
        advice.append("Limit coffee consumption to avoid sleep disruption.")
    
    if not advice:
        advice.append("Keep up your healthy lifestyle!")
    return advice
