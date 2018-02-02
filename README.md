# Soft Computing - Džunja Dejan RA52-2014

Instrukcije za pokretanje programa.

Ova komanda pokreće analizu svih video klipova, štampa rezultate u `out.txt` i pokreće `test.py`. Ukoliko ne postoji fajl `model.h5` u kom se nalazi NM, porkreće se treniranje NM na osnovu MNIST trening seta.

`python program.py all`

Ukoliko želite da ponovite treniranje NM, možete obrisati `model.h5` ili pokrenuti sledeću komandu.

`python trainClassifier.py`

Ukoliko želite da testirate samo jedan od video klipova, pokrenite sledeću komandu.

`python program.py x`

Umesto `x` upišite broj u intervalu `0-9`.

Primer `python program.py 5` pokreće analizu video klipa sa nazivom `video-5.avi`

Link [test_podaci](https://drive.google.com/drive/folders/0B1ZJXQY32LBUU3FiTS14a3NZd1U)

