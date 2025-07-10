import joblib
from pathlib import Path
from django.shortcuts import render
from .forms import PatientForm

# Caminho do modelo .pkl
MODEL_PATH = Path(__file__).resolve().parent / "model" / "rf_model.pkl"
model = joblib.load(MODEL_PATH)

# Ordem exata das variáveis usadas no treino
COL_ORDER = [
    "GENDER", "AGE", "SMOKING", "YELLOW_FINGERS", "ANXIETY", "PEER_PRESSURE",
    "CHRONIC_DISEASE", "FATIGUE", "ALLERGY", "WHEEZING", "ALCOHOL_CONSUMING",
    "COUGHING", "SHORTNESS_OF_BREATH", "SWALLOWING_DIFFICULTY", "CHEST_PAIN"
]

# Traduções dos campos para português
PT_LABELS = {
    "GENDER": "Gênero",
    "AGE": "Idade",
    "SMOKING": "Fuma",
    "YELLOW_FINGERS": "Dedos amarelados",
    "ANXIETY": "Ansiedade",
    "PEER_PRESSURE": "Pressão dos colegas",
    "CHRONIC_DISEASE": "Doença crônica",
    "FATIGUE": "Fadiga",
    "ALLERGY": "Alergia",
    "WHEEZING": "Chiado no peito",
    "ALCOHOL_CONSUMING": "Consumo de álcool",
    "COUGHING": "Tosse",
    "SHORTNESS_OF_BREATH": "Falta de ar",
    "SWALLOWING_DIFFICULTY": "Dificuldade para engolir",
    "CHEST_PAIN": "Dor no peito"
}

def index(request):
    probability = None
    respostas = None  # lista de (pergunta, resposta)

    if request.method == "POST":
        form = PatientForm(request.POST)
        if form.is_valid():
            # valores na mesma ordem do modelo
            data = [int(form.cleaned_data[field]) for field in COL_ORDER]

            # probabilidade (%) de classe 1
            probability = round(model.predict_proba([data])[0][1] * 100, 2)

            # monta lista vertical das respostas traduzidas
            respostas = []
            for field, value in zip(COL_ORDER, data):
                pergunta = PT_LABELS.get(field, field)
                if field == "AGE":
                    resposta = value
                else:
                    resposta = "Sim" if value == 1 else "Não"
                respostas.append((pergunta, resposta))
    else:
        form = PatientForm()

    context = {
        "form": form,
        "probability": probability,
        "respostas": respostas,
    }
    return render(request, "core/index.html", context)


