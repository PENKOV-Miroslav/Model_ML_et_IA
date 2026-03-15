import random
from mlops_tp.inference import InferenceModel

model = InferenceModel()

for _ in range(1000):
    features = {
        "category": random.choice(["Web Development", "Design", "Writing"]),
        "budget_usd": random.randint(100, 2000),
        "duration_days": random.randint(5, 40),
        "num_applicants": random.randint(1, 40),
        "freelancer_rating": round(random.uniform(2.0, 5.0), 1),
        "completion_time_days": random.randint(1, 40)
    }

    result = model.predict_with_details(features)

    if "proba" in result:
        p = result["proba"]["True"]
        if 0.45 < p < 0.55:
            print("Features trouvées :")
            print(features)
            print("Résultat :")
            print(result)
            break
else:
    print("Aucun cas proche de 50/50 trouvé.")