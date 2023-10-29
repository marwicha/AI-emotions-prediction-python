# Speech Emotion Recognition

L'objectif de ce projet est de mettre en production un modèle de détection de sentiment au travers de la voix.

## Utilisation


Il faut alors préciser deux données:
- **waveform**: liste des float du signal de la voix.
- **sample_rate**: fréquence d'échantillonnage du signal.

Ou alors il est aussi possible d'envoyer le fichier .wav en base64 en précisant seulement 
- **base64_waveform**: contenu du fichier .wav du signal encodé en base64.

L'API retourne alors les prédictions d'émotion sous format JSON comme l'exemple suivant:

```json
{
  "happy": 0.6809564232826233,
  "other": 0.15991879999637604,
  "angry": 0.1479712277650833,
  "sad": 0.006988084875047207,
  "neutral": 0.004165589343756437
}
```

Exemple d'utilisation de l'API côté client en python:

```python
def call_speech_recognition_api(url, waveform, sample_rate):
    # post data to the API
    data = {
        'waveform': list(waveform.astype('float64')),
        'sample_rate': sample_rate
    }
    r = requests.post(url, json=data)

    # get the results if status code OK
    predictions = r.json() if r.status_code == 200 else {'Error status code': r.status_code}
    # sort the emotions given their probabilities
    predictions = dict(sorted(predictions.items(), key=lambda x: x[1], reverse=True))
    return predictions

```
## Intsallation

```
cd ./dossier
```

Build du container docker en exécutant la ligne de commande située dans le fichier `docker_build.txt`

```
docker build -t project-name .
```

Run du container docker en exécutant la ligne de commande située dans le fichier `docker_run.txt`

```
docker run -d -p 5000:5000 -t project-name
```
