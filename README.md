OVERSIKT over filer som kan kjøres:

Sørg for at alle filene er i samme path, ettersom de arbeider med hverandre. 

"data_exploration.py" - utforsker de råe dataene som er hentet fra Statens Vegvesen og fra Geofysisk Institutt til UiB. Programmet outputter et komplett dataset som er allerede lagt inn her som "komplett_data.csv". Gir også ut "2023.csv" som er et likt dataset som "komplett_data.csv", men mangler target kolonnenen. Prediksjonene får disse dataene kommer i "INF161project.py". 

"INF161project.py" - kjører hele prosjektet. Denne utfører hyperparameter-tuning, så den kan ta en stund å kjøre. Spytter ut den beste modellen som blir kjørt på 2023 dataen. 

"INF161project.pdf" - rapport om hele prosjektet. 

"app.py" - lager en nettside der man kan skrive inn ulike vær- og tidsdata og få ut et predikert antall sykler. 
