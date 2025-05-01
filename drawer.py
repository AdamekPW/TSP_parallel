import matplotlib.pyplot as plt

def drawResults(nazwa_pliku, rodzaj):
    punkty_x = []
    punkty_y = []
    path = []

    with open(f"Benchmarks/{nazwa_pliku}", 'r') as plik:
        n = int(plik.readline()) 
        
        for i in range(n):
            linia = plik.readline()
            dane = linia.split()

            x = float(dane[1])  
            y = float(dane[2]) 
            
            punkty_x.append(x)
            punkty_y.append(y)
    
    with open(f"Results/{rodzaj}.txt", 'r') as plik:
        score = float(plik.readline()) 
        n = int(plik.readline())
        
        for i in range(n):
            linia = plik.readline()
            path.append(int(linia))


    plt.scatter(punkty_x, punkty_y, color='blue')

    for i in range(len(path)):
        x1 = punkty_x[path[i]]
        y1 = punkty_y[path[i]]

        x2 = punkty_x[path[(i+1) % len(path)]]
        y2 = punkty_y[path[(i+1) % len(path)]]

        plt.plot([x1, x2], [y1, y2], 'r-', alpha=0.5)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(nazwa_pliku)
    plt.grid(True)
    plt.show()

def drawTimes():
    pass

def calcSpeed():
    files = {'standard' : -1, 'openMP' : -1, 'cuda' : -1}
    for filename in files:
        try:
            with open(f"Times/{filename}.txt", 'r') as plik:
                time = int(plik.readline()) 
                files[filename] = time
        except FileNotFoundError:
            print(f"File {filename} not found")
    
    d = 1000000

    print(f"Standard: {files['standard']/d:0.2f} ms ")
    
    if (files['openMP'] != -1):
        openMP_speed = (files['standard'] / files['openMP']) 
        print(f"OpenMP: {files['openMP']/d:0.2f} ms | It is {openMP_speed:0.2f} speed of standard version")

    if (files['cuda'] != -1):
        cuda_speed = (files['standard'] / files['cuda']) 
        print(f"Cuda: {files['cuda']/d:0.2f} ms  | It is {cuda_speed:0.2f} speed of standard version")    
       



import sys;

rodzaj = ""
filename = ""

if len(sys.argv) <= 1:
    print("Nie podano argumentu.")
    exit(-1)

if len(sys.argv) > 1:
    type = sys.argv[1]
if len(sys.argv) > 2:
    rodzaj = sys.argv[2]    
if (len(sys.argv) > 3):
    filename = sys.argv[3]

if (type == '-t'):
    calcSpeed()
elif (type == '-r'):
    if (rodzaj == ""):
        print("Nie podano rodzaju kodu")
        exit(-1)
    if (filename == ""):
        print("Nie podano nazwy pliku benchmarku")
        exit(-1)
    drawResults(filename, rodzaj)
# readAndShow('berlin52.txt')