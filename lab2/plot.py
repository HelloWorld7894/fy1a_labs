import matplotlib.pyplot as plt
import numpy as np

# Tvoje data (pozor na ten poslední neúplný řádek)
data = [
    [0, 0.770850, 0.752280],
    [1, 0.771400, 0.763600],
    [6, 0.774290, 0.775660],
    [8, 0.774720, 0.778240],
    [6.6, 0.774090, 0.775630],
    [6.4, 0.774160, 0.775300]
]

# Bezpečnější extrakce dat (přeskočí řádky, kde chybí y1 nebo y2)
x_coords = np.array([row[0] for row in data if len(row) >= 3])
y1_points = np.array([row[1] for row in data if len(row) >= 3])
y2_points = np.array([row[2] for row in data if len(row) >= 3])

# Výpočet fitů
z1 = np.polyfit(x_coords, y1_points, 1) # Tau h
p1 = np.poly1d(z1)

z2 = np.polyfit(x_coords, y2_points, 1) # Tau d
p2 = np.poly1d(z2)

# --- VÝPOČET PRŮSEČÍKU ---
# z[0] je sklon (a), z[1] je posun (b) -> y = ax + b
a1, b1 = z1
a2, b2 = z2

if a1 == a2:
    print("Přímky jsou rovnoběžné, průsečík neexistuje.")
    intersect_x, intersect_y = None, None
else:
    intersect_x = (b2 - b1) / (a1 - a2)
    intersect_y = p1(intersect_x) # Dosadíme x do jakékoliv z rovnic

    print("-" * 30)
    print(f"Rovnice Tau h: y = {a1:.6f}x + {b1:.6f}")
    print(f"Rovnice Tau d: y = {a2:.6f}x + {b2:.6f}")
    print(f"PRŮSEČÍK: [x: {intersect_x:.4f}, y: {intersect_y:.4f}]")
    print("-" * 30)

# Vykreslení
plt.figure(figsize=(10, 6))

plt.scatter(x_coords, y1_points, color='blue', label='Tau h (data)', marker='o')
plt.plot(x_coords, p1(x_coords), "b--", alpha=0.5, label='Tau h (fit)')

plt.scatter(x_coords, y2_points, color='red', label='Tau d (data)', marker='x')
plt.plot(x_coords, p2(x_coords), "r--", alpha=0.5, label='Tau d (fit)')

# Pokud existuje průsečík, prskneme ho do grafu
if intersect_x is not None:
    plt.plot(intersect_x, intersect_y, 'go', markersize=10, label='Průsečík')
    plt.annotate(f'[{intersect_x:.2f}, {intersect_y:.4f}]', 
                 xy=(intersect_x, intersect_y), 
                 xytext=(intersect_x + 0.5, intersect_y),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

plt.title('Hledání průsečíku Tau h a Tau d')
plt.xlabel('N otáček')
plt.ylabel('Tau values')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.show()