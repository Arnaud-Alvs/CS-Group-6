# 📦 Streamlit App Groupe des BG 
```
⬆️ (Nom de l'app)
```

Description de l'app:

Solution: Apportez aux utilisateurs une interface leur permettant de détérminer le trajet optimal pour aller jetter leur poubelle.

L'idée serait que les utilisateurs puissent donner leur poubelle, et le programme détérminerait le trajet optimal. 

Les API, se trouvent dans ce lien:  https://daten.stadt.sg.ch/explore/dataset/abfuhrdaten-stadt-stgallen/information/?disjunctive.gebietsbezeichnung&disjunctive.sammlung&disjunctive.strasse




## Demo App

Lien de l'app sur streamlit (a voir pour plus tard)

## GitHub Codespaces

Pour les codespaces (et plus particulièrement le travail en groupe) comment ça marche ? 

Premièrement, chacun travaillera sur une branche particulière. 
Deuxièmement, les codespaces ne se partagent pas et "work" locally, ce qui veut dire que chaque changement fais sur votre codespace doit être push a la branche correspondante et qu'à chaque fois que vous y retravaillez dessus il faut le mettre a jour (dans le cas ou des changements ont été éffctué par d'autre personne. 

Code pour commit:  
---  
git add .  
git commit -m "Implemented feature X"  
git push origin feature-name  

Code pour remettre le code a niveau:  
---  
git checkout main  
git pull origin main  
git checkout feature-name    
git merge main  

Et voila, pour l'instant c'est tout ce que je sais.
## Infos Importantes
Ne pas toucher a la branch "main". 

