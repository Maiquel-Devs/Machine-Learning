from django import forms

# Ordem: 0=Não (padrão), 1=Sim
BINARY_CHOICES = [(0, "Não"), (1, "Sim")]

class PatientForm(forms.Form):
    GENDER = forms.ChoiceField(
        label="Gênero", choices=[(0, "Feminino"), (1, "Masculino")], initial=0
    )
    AGE = forms.IntegerField(label="Idade", min_value=1, max_value=120)

    SMOKING = forms.ChoiceField(label="Fuma", choices=BINARY_CHOICES, initial=0)
    YELLOW_FINGERS = forms.ChoiceField(label="Dedos amarelados", choices=BINARY_CHOICES, initial=0)
    ANXIETY = forms.ChoiceField(label="Ansiedade", choices=BINARY_CHOICES, initial=0)
    PEER_PRESSURE = forms.ChoiceField(label="Pressão dos colegas", choices=BINARY_CHOICES, initial=0)
    CHRONIC_DISEASE = forms.ChoiceField(label="Doença crônica", choices=BINARY_CHOICES, initial=0)
    FATIGUE = forms.ChoiceField(label="Fadiga", choices=BINARY_CHOICES, initial=0)
    ALLERGY = forms.ChoiceField(label="Alergia", choices=BINARY_CHOICES, initial=0)
    WHEEZING = forms.ChoiceField(label="Chiado no peito", choices=BINARY_CHOICES, initial=0)
    ALCOHOL_CONSUMING = forms.ChoiceField(label="Consumo de álcool", choices=BINARY_CHOICES, initial=0)
    COUGHING = forms.ChoiceField(label="Tosse", choices=BINARY_CHOICES, initial=0)
    SHORTNESS_OF_BREATH = forms.ChoiceField(label="Falta de ar", choices=BINARY_CHOICES, initial=0)
    SWALLOWING_DIFFICULTY = forms.ChoiceField(label="Dificuldade para engolir", choices=BINARY_CHOICES, initial=0)
    CHEST_PAIN = forms.ChoiceField(label="Dor no peito", choices=BINARY_CHOICES, initial=0)

