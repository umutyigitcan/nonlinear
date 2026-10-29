# AEROTRACE — Doğrusal Olmayan İHA Kontrol Laboratuvarı

Bu Python projesi, üç boyutlu hareketli bir hedefi izleyen İHA üzerinde dört doğrusal olmayan kontrol yöntemini aynı 60 saniyelik görevde karşılaştırır.

## Seçilen kontrolcüler

1. **Kazanç Ayarlı Doğrusal Olmayan PID:** Hata büyüklüğüne göre P ve D kazançlarını değiştirir; integral anti-windup ve kısmi rüzgâr telafisi kullanır.
2. **Backstepping:** Konum hatasından sanal hız komutu üretir, sonra gerçek hızın bu komutu kararlı biçimde izlemesini sağlar.
3. **Kayma Kipli Kontrol (SMC):** Model belirsizliği ve dış bozuculara dayanıklıdır. Titreşimi azaltmak için `sign` yerine yumuşak `tanh` sınır katmanı kullanılmıştır.
4. **Doğrusal Olmayan MPC (NMPC):** Kısa bir öngörü ufkunda birden fazla kontrol adayını doğrusal olmayan modelle simüle eder ve en düşük konum/hız/enerji maliyetine sahip adayı seçer.

## Model

- Durum: İHA konumu ve hızı (`x, y, z, vx, vy, vz`)
- Giriş: üç eksenli ivme komutu
- Doğrusal olmayan etkiler: karesel sürükleme, konum ve zamana bağlı rüzgâr, ivme doygunluğu
- Hedef: eliptik yatay hareket, yavaş sürüklenme ve iki frekanslı irtifa hareketi
- Süre: 60 saniye
- Örnekleme: 20 Hz (`dt = 0.05 s`)

## Çalıştırma

Python 3 ve NumPy yeterlidir. macOS ve çoğu standart Python kurulumunda Tkinter hazır gelir.

```bash
python3 -m pip install -r requirements.txt
python3 app.py
```

Sol panelden kontrolcüleri seçin ve **60 SN TESTİ BAŞLAT** düğmesine basın. Dört kontrolcü 2×2 görev ekranında eşzamanlı izlenir; sağ panelde X/Y/Z hataları, takip oranları ve anlık lider görünür. Test gerçek zamanlıdır ve sonuç ekranı tam 60 saniye tamamlandıktan sonra açılır.

## Sonuç ekranı

- X, Y ve Z hedef/İHA koordinatları
- X–Y rota karşılaştırması
- 3B rota izdüşümü
- Zamana bağlı konum hatası
- RMSE, maksimum hata, hedefi takip oranı, doğruluk puanı ve kontrol enerjisi
- Tüm kontrolcülerin hata ve X–Y rotalarının aynı grafikte karşılaştırılması
- P95 hata ve bileşik performans skoruyla otomatik kazanan özeti

Takip oranı, ilk 2 saniyelik warm-up sonrasında İHA ile hedef arasındaki mesafenin 0,5 metre veya altında kaldığı örneklerin yüzdesidir. Doğruluk puanı, mesafe hatasını 0–100 aralığına yumuşak biçimde dönüştürür.

## Test

```bash
python3 -m unittest -v test_simulation.py
```

> Bu model eğitim ve kontrolcü kıyaslama amaçlıdır. Gerçek uçuş için 6-DOF dinamikler, sensör modeli, gecikmeler, çarpışma önleme ve donanım emniyet sınırları eklenmelidir.
