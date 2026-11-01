# Vérification indépendante — préreg §§4–7

Calcul exécuté par `python3 codex/recompute.py` (NumPy 2.4.3 et bibliothèque standard), sans consulter ni importer le code du dépôt ; `data/` non modifié.
576/576 jobs uniques, même identité de run, aucun doublon ni job manquant : verdicts finaux pour le run relancé.
ρ₀ régénéré par seed, Dirichlet(1) successifs dans l’ordre phono, lex, syntax, sem ; TV = ½ somme absolue, puis maximum sur les espèces.

**Q1.** 12/60 marges utilisables ; 38 calculables dégénérées ; 10 absentes (48 sans marge utilisable). Aucun C2 à moins de 10⁻⁵ de l’égalité.
Lie : **non substituable** ; échecs C1 = 14, C2 = 2, C3 = 30 configurations. Sur 141 pas certifiés : 53 proven, 0 violated, 0 indeterminate, 88 missing.
Strang : **non substituable** ; échecs C1 = 8, C2 = 11, C3 = 25 configurations. Sur 162 pas certifiés : 103 proven, 0 violated, 0 indeterminate, 59 missing.
Listes des configurations en échec, ordre (α, γ, seed, h), point décimal :

- Lie C1 : (2, 20, 500, 0.05); (2, 20, 501, 0.05); (2, 30, 500, 0.05); (2, 30, 501, 0.05); (5, 20, 500, 0.05); (5, 20, 501, 0.05); (5, 30, 500, 0.05); (5, 30, 501, 0.05); (10, 10, 501, 0.05); (10, 20, 500, 0.05); (10, 20, 501, 0.05); (10, 30, 500, 0.05); (10, 30, 500, 0.025); (10, 30, 501, 0.05).
- Lie C2 : (5, 20, 500, 0.05); (10, 20, 501, 0.05).
- Lie C3 : (2, 0, 500, 0.025); (2, 0, 501, 0.025); (2, 5, 500, 0.025); (2, 5, 501, 0.025); (2, 10, 500, 0.025); (2, 10, 501, 0.025); (2, 20, 500, 0.025); (2, 20, 501, 0.025); (2, 30, 500, 0.025); (2, 30, 501, 0.025); (5, 0, 500, 0.025); (5, 0, 501, 0.025); (5, 5, 500, 0.025); (5, 5, 501, 0.025); (5, 10, 500, 0.025); (5, 10, 501, 0.025); (5, 20, 500, 0.025); (5, 20, 501, 0.025); (5, 30, 500, 0.025); (5, 30, 501, 0.025); (10, 0, 500, 0.025); (10, 0, 501, 0.025); (10, 5, 500, 0.025); (10, 5, 501, 0.025); (10, 10, 500, 0.025); (10, 10, 501, 0.025); (10, 20, 500, 0.025); (10, 20, 501, 0.025); (10, 30, 500, 0.025); (10, 30, 501, 0.025).
- Strang C1 : (2, 30, 500, 0.05); (10, 5, 501, 0.05); (10, 10, 500, 0.05); (10, 10, 501, 0.05); (10, 20, 500, 0.05); (10, 20, 501, 0.05); (10, 30, 500, 0.05); (10, 30, 501, 0.05).
- Strang C2 : (5, 5, 500, 0.05); (5, 10, 500, 0.05); (5, 20, 500, 0.05); (10, 0, 500, 0.05); (10, 0, 501, 0.05); (10, 5, 500, 0.05); (10, 5, 501, 0.05); (10, 10, 500, 0.05); (10, 10, 501, 0.05); (10, 20, 501, 0.05); (10, 30, 500, 0.025).
- Strang C3 : (2, 0, 500, 0.025); (2, 5, 500, 0.025); (2, 10, 500, 0.025); (2, 20, 500, 0.025); (2, 30, 500, 0.025); (5, 0, 500, 0.025); (5, 0, 501, 0.025); (5, 5, 500, 0.025); (5, 5, 501, 0.025); (5, 10, 500, 0.025); (5, 10, 501, 0.025); (5, 20, 500, 0.025); (5, 20, 501, 0.025); (5, 30, 500, 0.025); (5, 30, 501, 0.025); (10, 0, 500, 0.025); (10, 0, 501, 0.025); (10, 5, 500, 0.025); (10, 5, 501, 0.025); (10, 10, 500, 0.025); (10, 10, 501, 0.025); (10, 20, 500, 0.025); (10, 20, 501, 0.025); (10, 30, 500, 0.025); (10, 30, 501, 0.025).

Les deux échecs C2 de Lie sont des premiers pas non certifiés malgré une marge utilisable (§7, lignes 251–253) ; ceux de Strang satisfont e_X > s_S.

**Q2.** Lie : aucune sauvée. Strang : (5, 50, 500, 0.05), (5, 70, 500, 0.05), (5, 100, 500, 0.05) ; les neuf pas sont proven, aucun indeterminate.
**Q3.** Classes (first, second, neither, unresolved) : S = (0, 0, 20, 10), Lie = (0, 0, 9, 21), Strang = (0, 0, 16, 14).
Cellules admises : liste vide (0/30, seuil 15) ; **non identifiable** pour Lie et Strang (§7, lignes 273–279).
Classes par cellule : N = neither, U = unresolved ; chaque triplet est (S, Lie, Strang), présenté pour seed 500 / seed 501 :

- α=2 : γ=0 : NNN/NNN; γ=5 : NUN/NNN; γ=10 : NUN/NUN; γ=20 : NUN/NUU; γ=30 : UUU/UUU.
- α=5 : γ=0 : NNN/NNN; γ=5 : NUN/NNN; γ=10 : NUN/NUN; γ=20 : UUU/UUU; γ=30 : UUU/UUU.
- α=10 : γ=0 : NNN/NNN; γ=5 : NUU/NNN; γ=10 : UUU/NUU; γ=20 : UUU/NUU; γ=30 : UUU/UUU.

Médianes (a₁, a₂, pente), sur cellules résolues : S (−0.722852, −0.025733, −0.461433) ; Lie (−0.707187, −0.048992, −0.487482) ; Strang (0.443843, 3.159452, 1.801648).
Détails numériques reproductibles : [results.json](results.json) contient toutes les marges, comparaisons C2, états C3 par pas, classes, défauts, exposants et pentes par cellule.

**Comparaison au driver.** Aucun écart numérique ni de verdict avec les valeurs fournies ; les trois sauvées sont exactement celles annoncées, pour Strang.
Précision de libellé : « 38 dégénérées » compte seulement les marges calculables. Le « otherwise » du §5, lignes 175–180, inclut aussi les 10 absentes parmi les configurations sans marge utilisable ; le total est donc 48, sans effet sur les verdicts.
Déviation déclarée : le redémarrage après 263 job_error de code déroge au §4, lignes 147–154 (première observation, pas de relance), comme indiqué dans la note os.nice ; ces anciens enregistrements ne figurent pas dans le fichier fourni. Le présent contrôle porte sur le run relancé, sans audit du correctif.
Le host enregistré est macM3 (niceness 20), distinct du KXKM-AI/nice 19 prévu au §4, lignes 160–166 ; la note fournie renvoie à une autre note de déviation du host, non fournie.
SHA-256 de confirm.jsonl : `256c1d4196dcb32d9b5029dc7d4e7202a85d277996de5ca8c6a065f91598efed`.
