# gemini-qilgan-topshiriq
# 1-topshiriq: Kerakli kutubxonalarni loyihaga import qilish
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 2-topshiriq: O'rnatilgan kutubxonalarning versiyalarini tekshirib chiqish
print(f"NumPy versiyasi: {np.__version__}") # NumPy versiyasini konsolga chiqarish
print(f"Pandas versiyasi: {pd.__version__}") # Pandas versiyasini konsolga chiqarish

# 3-topshiriq: student_grades.csv faylini Pandas DataFrame'ga yuklash
df = pd.read_csv('student_grades.csv') # CSV faylni o'qib df o'zgaruvchisiga saqlash

# 4-topshiriq: Dastlabki va oxirgi 5 ta qatorni ko'rish
print("--- Dastlabki 5 qator (head) ---")
print(df.head()) # Dastlabki 5 ta qatorni konsolga chiqarish

print("\n--- Oxirgi 5 qator (tail) ---")
print(df.tail()) # Oxirgi 5 ta qatorni konsolga chiqarish

# 5-topshiriq: Ma'lumotlar turi, ustunlar va qatorlar sonini aniqlash
print("--- DataFrame haqida umumiy ma'lumot (info) ---")
df.info() # Har bir ustun turi va bo'sh bo'lmagan qiymatlar soni

print("\nDataFramening o'lchami (qatorlar, ustunlar):", df.shape) # Qator va ustunlar soni

# 6-topshiriq: Baholarning statistik ko'rinishini chiqarish
print("--- Statistik ko'rinish (describe) ---")
print(df.describe()) # Sonli ustunlarning mean, std, min, max va kvartillarini chiqarish

# 7-topshiriq: Har bir ustundagi bo'sh (NaN) qiymatlar sonini topish
missing_values = df.isnull().sum() # Har bir ustunda nechta bo'sh qiymat borligini hisoblash
print("Bo'sh qiymatlar soni:\n", missing_values)

# 8-topshiriq: Bo'sh 'grade' qiymatlarini fan bo'yicha o'rtacha baho bilan to'ldirish
# Har bir fan (subject) bo'yicha o'rtacha bahoni hisoblab, tegishli bo'sh joylarga joylash
df['grade'] = df.groupby('subject')['grade'].transform(lambda x: x.fillna(x.mean()))

# 9-topshiriq: Takrorlangan (duplicate) qatorlarni topish va olib tashlash
print("Takrorlangan qatorlar soni:", df.duplicated().sum()) # Takrorlangan qatorlar sonini sanash
df = df.drop_duplicates() # Takrorlangan qatorlarni o'chirib tashlash

# 10-topshiriq: 'grade' ustuni turini tekshirish va kerak bo'lsa float/int ga o'tkazish
df['grade'] = df['grade'].astype(float) # 'grade' ustunini tekshirib float turiga o'tkazish
print("'grade' ustunining ma'lumot turi:", df['grade'].dtype) # Ma'lumot turini tasdiqlash

# 11-topshiriq: 'grade' ustunini NumPy massiviga (array) o'tkazish
grades_array = df['grade'].to_numpy() # Pandas Series ob'ektini NumPy array'ga aylantirish

# 12-topshiriq: NumPy funksiyalari yordamida statistikani hisoblash
mean_val = np.mean(grades_array)   # O'rtacha qiymat
median_val = np.median(grades_array) # Mediana
std_val = np.std(grades_array)     # Standart og'ish
min_val = np.min(grades_array)     # Eng past baho
max_val = np.max(grades_array)     # Eng yuqori baho

print(f"O'rtacha baho: {mean_val:.2f}")
print(f"Mediana: {median_val:.2f}")
print(f"Standart og'ish: {std_val:.2f}")
print(f"Eng past baho: {min_val}")
print(f"Eng yuqori baho: {max_val}")

# 13-topshiriq: Har bir o'quvchining barcha baholari yig'indisini faqat NumPy orqali hisoblash
names_array = df['full_name'].to_numpy() # O'quvchilar ismlarini NumPy massiviga o'tkazish
unique_names = np.unique(names_array)    # Takrorlanmas o'quvchilar ro'yxatini olish

print("NumPy orqali o'quvchilar baholari yig'indisi:")
for name in unique_names:
    student_mask = (names_array == name) # Ushbu o'quvchiga tegishli qatorlarni belgilash (maska)
    student_sum = np.sum(grades_array[student_mask]) # Faqat ushbu o'quvchi baholarini qo'shish
    print(f"{name}: {student_sum:.2f}")

# 14, 15 va 16-topshiriqlar: O'quvchilarning o'rtacha bahosini hisoblash, yaxlitlash va saralash
student_means = df.groupby('full_name')['grade'].mean() # 14-topshiriq: Har bir o'quvchi bo'yicha o'rtacha baho
student_means = student_means.round(2)                 # 15-topshiriq: Natijani 2 xonagacha yaxlitlash
student_means = student_means.sort_values(ascending=False) # 16-topshiriq: Kamayish tartibida saralash

print("O'quvchilarning o'rtacha baholari (kamayish tartibida):")
print(student_means)

# 17-topshiriq: Eng yuqori o'rtacha bahoga ega 3 ta o'quvchini aniqlash
top_3 = student_means.nlargest(3) # Eng yuqori 3 ta ko'rsatkichni olish
print("Top 3 o'quvchi (Eng yuqori o'rtacha baho):\n", top_3)

# 18-topshiriq: Eng past o'rtacha bahoga ega 3 ta o'quvchini aniqlash
bottom_3 = student_means.nsmallest(3) # Eng past 3 ta ko'rsatkichni olish
print("Eng past 3 o'quvchi:\n", bottom_3)

# 19-topshiriq: Sinf bo'yicha eng yuqori va eng past yakka bahoni va uning egasini topish
max_grade_row = df.loc[df['grade'].idxmax()] # Eng yuqori baho joylashgan qator
min_grade_row = df.loc[df['grade'].idxmin()] # Eng past baho joylashgan qator

print(f"Eng yuqori yakka baho: {max_grade_row['grade']} | O'quvchi: {max_grade_row['full_name']} | Fan: {max_grade_row['subject']}")
print(f"Eng past yakka baho: {min_grade_row['grade']} | O'quvchi: {min_grade_row['full_name']} | Fan: {min_grade_row['subject']}")

# 20-topshiriq: Har bir fan bo'yicha o'rtacha bahoni hisoblash
subject_means = df.groupby('subject')['grade'].mean().round(2) # Fanlar bo'yicha o'rtacha baho
print("Fanlar bo'yicha o'rtacha baho:\n", subject_means)

# 21-topshiriq: O'quvchi–fan pivot jadvalini yaratish
pivot_df = df.pivot_table(index='full_name', columns='subject', values='grade') # O'quvchi va fan matritsasini tuzish
print("O'quvchi-Fan Pivot Jadvali:")
print(pivot_df)

# 22-topshiriq: Eng past natijali fanni aniqlash va izoh yozish
lowest_subject = subject_means.idxmin() # Eng past o'rtacha bahoga ega fan
lowest_score = subject_means.min()     # Shu fanning o'rtacha bahosi

print(f"Eng past ko'rsatkichga ega fan: {lowest_subject} ({lowest_score})")

# IZOH / SABAB TAXMINI:
# Eng past ko'rsatkich qayd etilgan fanda (masalan, Fizika yoki Matematika) o'quvchilar uchun mavzular
# murakkab bo'lishi, nazariy bilimlarni amalda qo'llashda qiyinchiliklar mavjudligi yoki baholash mezoni
# boshqa fanlarga qaraganda qat'iyroq ekanligi taxmin qilinadi.

# 23-topshiriq: O'quvchilarning o'rtacha bahosi bo'yicha bar chart
plt.figure(figsize=(10, 5)) # Grafik o'lchamini belgilash
student_means.plot(kind='bar', color='skyblue', edgecolor='black') # Ustunli diagramma chizish
plt.title("O'quvchilarning o me'yordagi o'rtacha baholari", fontsize=14) # Grafik sarlavhasi
plt.xlabel("O'quvchi ismi", fontsize=12) # X o'qi nomi
plt.ylabel("O'rtacha baho", fontsize=12) # Y o'qi nomi
plt.xticks(rotation=45) # X o'qidagi yozuvlarni qiyshaytirish
plt.grid(axis='y', linestyle='--', alpha=0.7) # Katakchalar (grid) qo'shish
plt.tight_layout() # Joylashuvni moslash
plt.show() # Grafikni ko'rsatish

# 24-topshiriq: Fanlar bo'yicha o'rtacha baho diagrammasi
plt.figure(figsize=(8, 5)) # Grafik o'lchamini belgilash
subject_means.plot(kind='bar', color='coral', edgecolor='black') # Ustunli diagramma chizish
plt.title("Fanlar bo'yicha o'rtacha baholar", fontsize=14) # Sarlavha
plt.xlabel("Fanlar", fontsize=12) # X o'qi nomi
plt.ylabel("O'rtacha baho", fontsize=12) # Y o'qi nomi
plt.xticks(rotation=0) # X o'qidagi yozuvlarni to'g'ri ko'rsatish
plt.grid(axis='y', linestyle='--', alpha=0.7) # Grid qo'shish
plt.tight_layout() # Joylashuvni moslash
plt.show() # Grafikni ko'rsatish

# 25 va 26-topshiriqlar: Baholarning umumiy taqsimot histogrammasi
plt.figure(figsize=(8, 5)) # Grafik o'lchami
plt.hist(df['grade'], bins=10, color='mediumseagreen', edgecolor='black', alpha=0.8) # Histogramma
plt.title("Sinf bo'yicha baholarning umumiy taqsimoti (Histogramma)", fontsize=14) # Sarlavha
plt.xlabel("Baholar", fontsize=12) # X o'qi
plt.ylabel("Chastota (O'quvchilar soni)", fontsize=12) # Y o'qi
plt.grid(axis='y', linestyle='--', alpha=0.7) # Grid
plt.tight_layout() # Joylashuv
plt.show() # Grafikni ko'rsatish

### 9-BOSQICH. Xulosa va Tavsiyalar

#### Tahlil Xulosasi:
1. O'tkazilgan tahlillar shuni ko'rsatadiki, sinfning umumiy o'rtacha o'zlashtirish ko'rsatkichi barqaror holatda.
2. O'quvchilar orasida yetakchi o'rinlarni egallagan Top-3 o'quvchi boshqa fanlar bilan birga barcha yo'nalishlarda yuqori natija ko'rsatgan.
3. Aksincha, o'rtacha bahosi eng past bo'lgan 3 nafar o'quvchi qo'shimcha yordamga muhtoj hisoblanadi.
4. Fanlar kesimida tahlil qilinganda, eng past o'rtacha ball aniqlangan fanda o'quvchilarning asosiy qismi qiyinchilikka uchragani ma'lum bo'ldi.
5. Umumiy baholar taqsimoti (histogramma) sinfda o'zlashtirish darajasi asosan o'rta va yuqori ballar oralig'ida toplanganini ko me'yorda ko'rsatmoqda.

#### Tavsiyalar:
1-tavsiya: Eng past ko'rsatkich qayd etilgan fan bo'yicha o'quvchilarning bo'shliqlarini to'ldirish uchun haftalik qo'shimcha amaliy mashg'ulotlar tashkil etish tavsiya etiladi.
2-tavsiya: O'zlashtirishi past bo'lgan 3 nafar o'quvchiga kuchliroq o'quvchilarni biriktirish (peer-learning) yoki ular uchun individual o'quv rejasini ishlab chiqish zarur.
