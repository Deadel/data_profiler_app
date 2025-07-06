# 📊 Data Profiler App — Interaktywna analiza CSV z ML i wizualizacją

Data Profiler App to lekka aplikacja webowa stworzona w Pythonie, która pozwala na:

- Wgrywanie plików CSV i szybki podgląd danych  
- Wyświetlanie statystyk opisowych i braków danych  
- Wizualizację korelacji w postaci heatmapy  
- Generowanie interaktywnych wykresów (histogramy, boxploty)  
- Trening prostego modelu klasyfikacyjnego (Random Forest) na wybranej kolumnie  
- Automatyczny zapis wytrenowanego modelu do pliku `.pkl` z timestampem  

---

## 🚀 Funkcjonalności

| Funkcja                      | Opis                                                                                   |
|-----------------------------|----------------------------------------------------------------------------------------|
| Upload CSV                  | Wczytaj lokalny plik CSV w interfejsie                                                 |
| Podgląd danych              | Wyświetlanie pierwszych wierszy tabeli i statystyk                                    |
| Braki danych                | Identyfikacja kolumn z brakującymi wartościami                                        |
| Heatmapa korelacji          | Wizualizacja korelacji między kolumnami numerycznymi                                  |
| Wykresy interaktywne        | Histogramy dla danych kategorycznych i boxploty dla danych liczbowych                  |
| Trening modelu ML           | Użycie Random Forest Classifier do klasyfikacji z automatycznym podziałem na train/test |
| Zapis modelu                | Model zapisywany jest automatycznie do folderu `models/` z unikalną nazwą             |

---

## 🧰 Technologie

- Python 3.8+  
- [Gradio](https://gradio.app/) – szybki UI webowy  
- Pandas – manipulacja danymi  
- Matplotlib / Seaborn – wizualizacje statyczne (heatmapa)  
- Plotly – wykresy interaktywne  
- Scikit-learn – model Random Forest  
- Joblib – serializacja modeli  

---

## 🏁 Jak uruchomić lokalnie?

1. **Zainstaluj wymagania:**

```bash
pip install -r requirements.txt


## 📁 Struktura projektu
data_profiler_app/
├── app.py              # główny plik aplikacji Gradio
├── utils.py            # funkcje do analizy danych i modelu ML
├── sample.csv          # przykładowy plik CSV do testów
├── requirements.txt    # lista pakietów Python
├── models/             # folder z zapisanymi modelami (tworzony automatycznie)
└── README.md           # ten plik


## 📝 Uwagi
Aplikacja oczekuje dobrze sformatowanego pliku CSV (separator ,, bez błędów formatowania).

Jeśli plik jest niepoprawny, UI wyświetli stosowny komunikat błędu.

Model ML trenuje się na wybranej kolumnie i zapisuje do folderu models/.

Nazwa pliku modelu zawiera znacznik czasu, np. model_20250706_113045.pkl.

## 💡 Pomysły na rozwój
Możliwość pobierania wytrenowanego modelu z UI

Obsługa regresji i innych modeli ML

Eksport raportów do PDF/HTML

Integracja z bazą danych lub chmurą

Dashboard z historią trenowań i metrykami

## 🧑‍💻 Autor - Marcin Kondracki
Projekt stworzony jako demonstracja umiejętności Python, analizy danych, ML oraz budowy prostych aplikacji webowych z interaktywnym UI.

## 📜 Licencja
MIT License – używaj i modyfikuj dowolnie.


