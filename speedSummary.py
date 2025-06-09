def wczytaj_dane_z_pliku(nazwa_pliku):
    with open(nazwa_pliku, 'r') as plik:
        zawartosc = plik.read()
        liczby = [float(liczba.strip()) for liczba in zawartosc.split(',') if liczba.strip()]
    return liczby


labels = wczytaj_dane_z_pliku('Times/x.txt')
standard = wczytaj_dane_z_pliku('Times/StandardTimes')
openMP = wczytaj_dane_z_pliku('Times/OpenMPTimes')
cuda = wczytaj_dane_z_pliku('Times/CudaTimes')


print("| n | Ts(n) | To(n) |Ts(n)/To(n)| Tc(n) | Ts(n) / Tc(n) |")
for i in range(len(standard)):
    print(f"| {labels[i]} | {standard[i]:.2f} | {openMP[i]:.2f} | {standard[i] / openMP[i]:.2f} | {cuda[i]:.2f} | {standard[i] / cuda[i]:.2f} |")