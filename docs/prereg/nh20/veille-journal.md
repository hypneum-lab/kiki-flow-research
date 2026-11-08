# Journal de veille — NH-20 (Lie contre Strang, splitting W₂/Fisher-Rao)

Un bloc par tour. Règles : identifiants résolus par API avant écriture (arXiv
avec un User-Agent neutre, 3 s entre appels ; Crossref `works/<doi>` ;
Semantic Scholar en lot par `s2.py`) ; « [résumé moteur] » pour ce qui n'a pas
été lu en entier ; chaque tour change de canaux par rapport au précédent.

## Tour 1 — 2026-11-01

**Point de départ.** Les six sources imposées par la spec (§5.1) et le plan
(Task 6), plus les veilles du 01/11 (thèse 3, axe R3) et le rapport
d'analyse (B7-B10, M10-M15, C2, C5).

**Canaux interrogés.** arXiv (API `export.arxiv.org/api/query`) ;
Crossref-journaux (API `works?query=`) ; thèses (notices ABES remontées par
Crossref, préfixe 10.70675 ; HAL non interrogé directement) ; code (API de
recherche de dépôts GitHub) ; remontée de citations amont et aval (Semantic
Scholar, un appel en lot, un 429 puis succès après 10 s).

**Requêtes exactes.**

| canal | requête | brut |
|---|---|---|
| arXiv | `all:"Strang splitting" AND all:Wasserstein` | 2 |
| arXiv | `all:splitting AND all:JKO` | 8 |
| arXiv | `all:"Wasserstein-Fisher-Rao" AND all:splitting` | 2 |
| arXiv | `all:"Hellinger-Kantorovich" AND all:splitting` | 0 |
| arXiv | `all:"second order" AND all:"Wasserstein gradient flow"` | 18 |
| arXiv | `all:"operator splitting" AND all:"gradient flow" AND all:"order of convergence"` | 0 |
| arXiv | `all:"entropic" AND all:"JKO" AND all:"convergence rate"` | 1 |
| arXiv | `all:"minimizing movement" AND all:"error estimate"` | 2 |
| Crossref | `Strang splitting Wasserstein gradient flow` | 20 |
| Crossref | `JKO splitting scheme order of convergence` | 20 |
| Crossref | `Wasserstein Fisher-Rao splitting scheme` | 20 |
| GitHub | `JKO splitting` ; `Wasserstein Fisher Rao splitting` ; `entropic JKO` ; `Strang splitting Wasserstein` | 0 ; 0 ; 3 ; 0 |
| S2 (citations) | graines arXiv:1602.04457, 1512.02783, 2501.14660, 2605.31369, 1506.00126, 1704.04541, 2002.03035 ; champs `references.*`, `citations.*` | 615 lignes |

**Comptes.** Brut : 33 (arXiv, distincts) + 60 (Crossref) + 3 (GitHub) + 615
(S2) = 711. Dédoublonné : ≈ 560 (496 identifiants distincts côté S2, 33 côté
arXiv, recouvrements non comptés finement). Filtre de titre sur la remontée
(split, Strang, second order, error estimate, rate, Fisher-Rao, Hellinger,
unbalanced, Trotter, BDF, midpoint) : 54 lignes. **Gardé : 14.**

**Entrées gardées (identifiant résolu ; statut de lecture).**

| id | auteurs, année | canal | statut | pourquoi |
|---|---|---|---|---|
| arXiv:2511.18060 | Crucinio, Pathiraja 2025 | arXiv + S2 (aval de 1602.04457) | lu en entier | Lie W-FR / FR-W avec sous-flots exacts, ordre des opérateurs ; antériorité la plus proche |
| arXiv:2109.02792 | Liu, Wang, Wang 2021 | S2 (aval) | lu en entier | Strang d'ordre 2, réaction-diffusion à énergie commune, sous-pas d'ordre 2 |
| arXiv:1802.08089 | Wibisono 2018 | S2 (amont de 2002.03035) | lu en entier (p. 1-8, annexe B) | symétrisation, adjoint, ordre 2 ; formes closes gaussiennes |
| arXiv:1005.0998 | Clément, Maas 2010 | veille du 01/11 | lu en entier | Trotter métrique, convergence sans taux |
| arXiv:2410.03367 | Cancès, Monsaingeon, Natale 2024 | veille du 01/11 | lu (p. 1-3, §5) | JKO ordre 1 ; référence d'ordre 2 |
| arXiv:2511.14827 | Halmos, Hanin 2025 | arXiv | [résumé moteur] | énergie modifiée du JKO à l'ordre η² : prédirait le défaut de Strang-JKO |
| arXiv:2604.08283 | Baradat, Cherf 2026 | arXiv | [résumé moteur] | taux du JKO entropique, ε = ατ |
| arXiv:2511.10884 | Chu, Jacobs 2025 | arXiv | [résumé moteur] | schéma W₂ d'ordre 2 accéléré prouvé |
| arXiv:2209.14622 | Gallouët, Natale, Todeschi 2022 | arXiv + S2 | [résumé moteur] | BDF2 variationnel W₂, ordre 2 numérique |
| arXiv:2105.11146 | Adams, Duong, dos Reis 2021 | arXiv + S2 | [résumé moteur] | splitting transport exact / JKO, version entropique |
| arXiv:1606.04793 | Carlier, Laborde 2016 | S2 (aval de 1506.00126) | [résumé moteur] | splitting transport sans divergence / JKO semi-implicite |
| arXiv:2608.25279 | Bou-Rabee 2026 | arXiv | [résumé moteur] | Strang de Langevin cinétique (autre objet ; cité comme seule occurrence « Strang ») |
| DOI 10.1093/imanum/drag053 | Hraivoronska, Santambrogio 2026 | Crossref | titre et métadonnées | JKO entièrement discret (grille) : pertinent pour C3 |
| DOI 10.70675/5ca4e714zf1c1z4f17za507z56a8e3bb0960 ; DOI 10.70675/97afc817z20a4z4ea5z9eadz6b24b7abf0f1 | thèses Laborde ; Caillet (ABES) | Crossref (thèses) | titre seul | thèses françaises sur JKO, à ouvrir au tour 2 (HAL) |

Code : `gpeyre/2015-SIIMS-wasserstein-jko` (code de référence du JKO entropique
de Peyré 2015) retenu comme source de nature « code » pour le tour 2 ; non
lu à ce tour.

**Écartés (motif).** Applications de flots W₂ sans splitting ni ordre
(génératif, apprentissage, épidémiologie) ; splittings hors W₂ (Navier-Stokes,
MHD, poroélasticité) ; résultats Crossref sans rapport (photocatalyse,
psychiatrie).

**Ce que le tour a changé dans la note.**

1. La question ouverte se resserre : Lie avec sous-flots exacts pour WFR est
   publié (2511.18060) ; Strang d'ordre 2 avec sous-pas d'ordre 2 pour une
   réaction-diffusion à énergie commune aussi (2109.02792). Reste l'ordre avec
   sous-pas JKO, l'effet de ε(h) et de la grille, et l'appariement adjoint.
2. Ajout de la condition C5 (nature des sous-pas) et de l'option
   « symétrique-adjoint » (1802.08089).
3. Ajout de la condition C2 (ε = s² injecte un défaut ∝ h² dans
   l'auto-convergence) et nuance au rapport du 01/11 sur Carlier 3.16 (l'écart
   est d'un facteur 18 à ε = 0,01, mais seulement logarithmique à ε = h²).
4. Référence exacte disponible pour le dessin minimal : solution du flot WFR
   gaussien (2511.18060, éq. 3.1-3.2).

**Prochain tour (canaux différents).** OpenReview (ICLR 2027, ateliers
NeurIPS 2026 : « Strang », « Wasserstein-Fisher-Rao », « JKO splitting ») ;
HAL en direct (thèses Laborde, Caillet, Natale ; « schéma de splitting
JKO ») ; Zenodo et Hugging Face (code ou données de splitting WFR) ; bioRxiv
pour la forme ; remontée de citations sur 2511.18060, 2109.02792, 1802.08089 ;
lecture intégrale de 2511.14827 (Halmos-Hanin).

## Tour 2 — 2026-11-01

**Point de départ.** Note v0 (tour 1) et ses marques « à recouper » ; canaux non
encore interrogés : OpenReview, bioRxiv, HAL et theses.fr en direct, code et
données (GitHub, Hugging Face, Zenodo, OSF), plus une remontée de citations sur
les sources du tour 1 laissées sans graine. Prédiction de la note à tester
contre la littérature : « Strang avec sous-pas JKO reste d'ordre 1 ».

**Canaux interrogés.** OpenReview (API v2 `notes/search`, filtre de date posé
après coup : 2024-01 et plus) ; HAL (API `search`, `docType_s:THESE`) ;
theses.fr (API `v1/theses/recherche`) ; GitHub (`gh search repos` puis
`gh search code`) ; Hugging Face (API `models` et `datasets`) ; Zenodo (API
`records`, `sort=mostrecent`) ; OSF (API `registrations`) ; bioRxiv (API
`details` sondée, puis Europe PMC en remplacement, voir ci-dessous) ;
remontée de citations Semantic Scholar (`citation_snowball.py`, deux lots).
User-Agent neutre « hypneum-lab reading-gate » ; 3 s entre deux appels arXiv ;
un seul 429 (OpenReview, requête relancée avec succès 20 s plus tard).

**Requêtes exactes.**

| canal | requête | brut |
|---|---|---|
| OpenReview | `JKO scheme` ; `Wasserstein gradient flow splitting` ; `Wasserstein-Fisher-Rao` ; `Strang splitting Wasserstein` ; `Fisher-Rao splitting` | 50 ; 50 ; 50 ; 50 ; 50 |
| OpenReview | `Hellinger-Kantorovich` ; `Lie-Trotter splitting gradient flow` ; `operator splitting Wasserstein Fisher-Rao` (429 puis relance) ; `second-order JKO` | 11 ; 50 ; 50 ; 50 |
| OpenReview | `order of convergence JKO scheme` ; `unbalanced optimal transport gradient flow splitting scheme` ; `entropic JKO regularization` ; `Strang splitting gradient flow measures` | 50 ; 50 ; 50 ; 50 |
| OpenReview | `Wasserstein Fisher Rao Gradient Flows: Operator Splitting Convergence Speed` (titre trouvé dans la remontée S2) | 10 |
| HAL (thèses) | `JKO` ; `"Wasserstein-Fisher-Rao"` ; `"Hellinger-Kantorovich"` ; `Wasserstein splitting` ; `"flot de gradient" Wasserstein` | 9 ; 0 ; 0 ; 8 ; 9 |
| HAL (thèses) | `"schéma de splitting" transport optimal` ; `Scharfetter-Gummel Wasserstein` ; `"splitting de Strang" Wasserstein` ; `"Jordan-Kinderlehrer-Otto"` ; `"minimizing movement"` | 1 ; 0 ; 0 ; 1 ; 0 |
| theses.fr | `JKO` ; `Wasserstein Fisher-Rao` ; `splitting Wasserstein` ; `flot de gradient Wasserstein` | 10 ; 0 ; 9 ; 37 (50 max par page) |
| theses.fr | `schéma de splitting` ; `transport optimal régularisation entropique` ; `Hellinger-Kantorovich` ; `équations de réaction-diffusion transport optimal` | 50 sur 252 ; 30 ; 0 ; 18 |
| GitHub (dépôts) | `JKO scheme` ; `Wasserstein Fisher-Rao` ; `JKO splitting` ; `unbalanced optimal transport gradient flow` ; `Strang splitting Wasserstein` ; `Wasserstein gradient flow JKO` | 2 ; 7 ; 0 ; 0 ; 0 ; 0 |
| GitHub (dépôts) | `jko wasserstein gradient flow` ; `wasserstein-fisher-rao` ; `entropic jko` ; `hellinger-kantorovich` ; `lie-trotter wasserstein` | 0 ; 8 ; 0 ; 2 ; 0 |
| GitHub (code) | `WFR splitting` ; `strang repo:<dépôt retenu>` | 5 fichiers ; 1 |
| Hugging Face | `JKO` ; `Wasserstein-Fisher-Rao` ; `wasserstein gradient flow` (modèles et jeux de données) | 60 (bruit : préfixes « jko… ») ; 0 ; 0 |
| Zenodo | `JKO` ; `"Wasserstein-Fisher-Rao"` ; `"JKO scheme" splitting` ; `"Strang splitting" Wasserstein` ; `"Hellinger-Kantorovich" splitting` ; `"unbalanced optimal transport" "gradient flow"` | jusqu'à 25 par requête (130 notices en tout, 104 distinctes) |
| OSF | `JKO` ; `Wasserstein-Fisher-Rao` ; `Wasserstein gradient flow` ; `splitting Strang` | 0 ; 0 ; 0 ; 0 |
| bioRxiv | `api.biorxiv.org/details/biorxiv/2026-10-25/2026-11-01/0` (sonde) | 1 641 préprints en 7 jours |
| Europe PMC | résumé contient `Wasserstein-Fisher-Rao` ou `JKO scheme` ou `Strang splitting` ou `Hellinger-Kantorovich` ou `Jordan-Kinderlehrer-Otto`, éditeur bioRxiv, 2024-01-01 à 2026-11-01 ; idem `Wasserstein gradient flow` ET `splitting` sur les préimpressions ; témoin : `Wasserstein`, éditeur bioRxiv | 0 ; 0 ; témoin 66 |
| S2 (citations) | lot A : graines arXiv:2511.18060, 2109.02792, 1802.08089, 1005.0998, 2410.03367 ; lot B : arXiv:2502.12666, 2604.08283, 2511.14827, 2511.10884 (lot B ajouté après lecture) | 509 ; 75 |

**Écart au brief, bioRxiv.** L'API `details` de bioRxiv n'offre aucun filtre par
mots-clés (1 641 préprints sur la seule semaine du 25/10 au 01/11 : un balayage
2024-2026 demanderait de l'ordre de dix mille pages de cent notices). Remplacé
par Europe PMC, qui indexe bioRxiv avec recherche dans les résumés ; requête
témoin non vide (66 notices « Wasserstein »), dont une seule touche l'objet du
filtre (« unbalanced », AtlasOT, Gromov-Wasserstein, hors sujet). Zéro retenu.

**Comptes.** Brut : OpenReview 621 ; HAL 28 ; theses.fr 154 (356 résultats
cumulés, pages de 50) ; GitHub 19 dépôts ; Hugging Face 60 ; Zenodo 130 ;
OSF 0 ; Europe PMC 0 ; S2 584 lignes ; soit 1 596. Dédoublonné : OpenReview
428 ; HAL 19 ; theses.fr 143 ; GitHub 12 ; Hugging Face 60 ; Zenodo 104 ; S2
539 identifiants distincts dont 471 absents du tour 1 ; soit 1 305. Filtre
par titre puis par résumé (JKO, Wasserstein-Fisher-Rao, Hellinger-Kantorovich,
splitting, Strang, Trotter, ordre, second ordre, ε) ; les lignes qui passent
le filtre ont été ouvertes une à une (pas de décompte tenu).
**Gardé : 24** (OpenReview 4 ; thèses 5 ; code 1 ; articles 14), dont 12 lus
en tout ou partie ce tour (PDF ou dépôt ouvert), 1 lu au tour 1 (2109.02792),
11 en [résumé moteur], métadonnées ou titre. Hugging Face, Zenodo, OSF,
Europe PMC et la plupart des dépôts GitHub n'ont rien donné : zéro retenu.

**Entrées gardées (identifiant résolu par API ; statut de lecture).**

| id | auteurs, année | canal | statut | ce que ça change |
|---|---|---|---|---|
| arXiv:2502.12666 (= DOI 10.1137/25m1736001, SIAM J. Math. Anal. 2026) | Baradat, Hraivoronska, Santambrogio 2025 | références de 2604.08283 (hors graines S2) | lu p. 1-10 (intro, déf. 2.1-2.3, Th. 2.9, Rem. 2.10) | **dépasse** : JKO entropique avec ε/τ → α, diffusion α/2 Δρ ; pour α = 0, plus besoin de ε\|log ε\| ≤ Cτ² (tore) |
| arXiv:2604.08283 | Baradat, Cherf 2026 | arXiv (tour 1) | lu p. 1-4 (promu de [résumé moteur]) | confirme ε = ατ ; borne W₂ ≤ C(√α + 1/√n) (Cor. 1.3.1) |
| arXiv:2511.14827 (v3, 03/03/2026) | Halmos, Hanin 2025 | arXiv (tour 1) | lu p. 1-8, recherche de mots-clés sur les 53 p. (aucun « splitting », « Strang », « Trotter », « composition ») | **outil** : énergie modifiée J^η = J − (η/4)\|∂J\|² à O(η²) ; ne traite pas la composition |
| arXiv:2511.10884 | Chu, Jacobs 2025 | arXiv (tour 1) ; jumeau OpenReview cJk5TcGdd1 | lu p. 1-5 (promu de [résumé moteur]) | **confirme** : premier ordre 2 prouvé (schéma trapézoïdal lagrangien, hors JKO) ; les autres schémas d'ordre 2 sans taux ou en O(√τ) |
| arXiv:2209.14622 (= DOI 10.1090/mcom/3951, Math. Comp. 2024) | Gallouët, Natale, Todeschi 2022 | arXiv (tour 1) | lu p. 1-3 + tableau 1 (promu) | BDF2 : convergence prouvée, ordre 2 numérique ; recoupe la ligne 10 |
| arXiv:2105.11146 (= DOI 10.3934/dcds.2022109, DCDS 2022) | Adams, Duong, dos Reis 2021 | arXiv (tour 1) | lu résumé, §1, §4 (hyp. 4.1, Th. 4.2) (promu) | splitting transport exact / JKO : convergence sans ordre ; reprend ε\|log ε\| ≤ Ch² |
| arXiv:2105.09415 (= DOI 10.1137/21m1421283, SINUM 2022) | Liu, Wang, Wang 2021 | S2 (aval de 2109.02792) | lu résumé et intro | Lie (variational operator splitting) : O(Δt) ; pendant du Strang de 2109.02792 |
| arXiv:2109.02792 (= DOI 10.1137/21m1444825, SISC 2022) | Liu, Wang, Wang 2021 | tour 1 | déjà lu ; version revue résolue | lève « à recouper » (ligne 5) |
| arXiv:2401.01722 (= DOI 10.1017/s0962492923000077, Acta Numerica 2024) | Blanes, Casas, Murua 2024 | S2 (amont de 2109.02792) | lu §1.1-1.3 (p. 1-11) | **confirme** : symétrique ⇒ ordre pair ; Euler implicite = adjoint de l'explicite ; remplace Strang 1968 (métadonnées) |
| arXiv:1711.02935 (= DOI 10.1051/m2an/2018045, M2AN 2019) | Matthes, Plazotta 2017 | S2 (lot B) | [résumé moteur] | BDF2 variationnel : ordre ½ prouvé en général, 2 attendu en régulier |
| DOI 10.1016/J.CRMA.2017.02.001 | Legendre, Turinici 2017 | S2 (amont de 2410.03367) ; citée par 2209.14622 | métadonnées Crossref | autre variante d'ordre 2 de JKO |
| arXiv:2211.07011 | Han, Esedoglu, Garikipati 2022 | S2 (lot B) | [résumé moteur] | schémas d'ordre 2-3 stables en énergie, flots W₂ en exemples |
| arXiv:2306.06940 | Malamut, Sylvestre 2023 | S2 (lot B) | [résumé moteur] | vitesse de convergence en ε du transport régularisé (\|γ_ε − γ₀\| ≳ √ε) |
| DOI 10.1137/17M1158379 ; arXiv:1711.05434 | Carrillo, Fagioli, Santambrogio, Schmidtchen 2017 | S2 (amont de 2410.03367) | [résumé moteur] | splitting EDO + transport optimal : existence, pas d'ordre |
| thèse HAL tel-03500566 (theses.fr 2021UPSLD036) | Todeschi 2021 | theses.fr | lu chap. 4 §4.6 et chap. 5 §5.1 (p. 108-112 imprimées) | « Le JKO est naturellement d'ordre un » ; BDF2 modifié d'ordre 2 numérique |
| thèse HAL tel-01510949 (2016PSLED014) | Laborde 2016 | HAL | lu §7.2 (p. 122-123 imprimées ; reste du chapitre parcouru par mots-clés) | splitting WFR « travail en préparation avec Gallouët et Monsaingeon » : existence, pas d'ordre |
| thèse HAL tel-01881166 (2017PSLED063) | Chizat 2017 | HAL | lu §6 (comparaison de splitting, p. PDF 194) | « nous ne savons pas prouver la vraie convergence » pour Φ non linéaire |
| theses.fr s409481 | Lin, direction Santambrogio, « Flots de gradient d'ordre supérieur dans l'espace de Wasserstein » (en cours) | theses.fr | titre | thèse sur l'ordre supérieur, pas encore déposée |
| theses.fr s409961 | Cherf, direction Santambrogio, « Problème de Muskat et régularisation entropique du schéma de JKO » (en cours) | theses.fr | titre | prolongement de 2604.08283 |
| OpenReview cJk5TcGdd1 | titre identique à Chu-Jacobs | OpenReview | titre et résumé | jumeau de 2511.10884 ; aucune nouveauté |
| OpenReview pmRC8xTiym (ICLR 2027, soumission) | anonymes | OpenReview | [résumé moteur] ; PDF refusé (403), **non vérifié** ; pas de version arXiv de même titre | « WFR splitting scheme », convergence exponentielle vers la gaussienne optimale ; optimisation, pas d'ordre annoncé |
| OpenReview rPgc5brxmT (NeurIPS 2024) ; QoagfaT4tZ (CORR 2024) | Interaction-Force Transport ; Fisher-Rao + splitting d'opérateurs | OpenReview | [résumé moteur] | algorithmes à splitting, garanties de convergence exponentielle, pas d'ordre |
| code : `zheyuanlai/affine-invariant-Gaussian-gradient-flow` (sans licence, 0 étoile, poussé 2026-08-21) | — | GitHub (`gh search code`) | lu `README`, `src/wfr_gradient_flow/methods.py` (175 l.), `reports/wfr_gradient_flow_report.tex:419-421` | **outil + preuve d'absence** : demi-pas W gaussien en forme close, splitting de Lie W puis FR ; rapport : « Lie-Trotter ; un Strang … n'est pas testé » |

**Écartés (motif).** Hugging Face, Zenodo, OSF, Europe PMC : bruit sans lien
(préfixes « jko… », physique, économie, Gromov-Wasserstein). OpenReview :
modèles génératifs et échantillonnage utilisant WFR sans analyse d'ordre
(flow matching, VI, politique), `bCNgWCHoSn` (certificats JKO neuronaux,
retiré avant évaluation), `EbE2zyPxAc` (JKO à moment, convergence en pas → 0
sans taux). Thèses ouvertes sans aucune occurrence de « Strang » ou de
splitting de flot : Caillet 2025 (tel-05262724), Coudreuse 2026
(tel-05798816), Séjourné 2022 (tel-05018660) ; Wu-Wang 2501.06408 (JKO
statistique). Autres dépôts GitHub : plongements de requêtes, Hawkes, distances
WFR sans flot.

**Saturation (remontée de citations).** Graines : 9 (5 lot A + 4 lot B).
S2 : 584 lignes brutes (575 après fusion des deux lots), 539 identifiants distincts, dont **471 absents du tour 1**
(`snowball-tour2.tsv`, colonne `lot`). Pertinents parmi ces 471, titre lu :
**14** (lot A : Legendre-Turinici, Liu-Wang-Wang JCP 2021 et 2105.09415,
Carrillo et al. 2018, Blanes et al., Kato 1978, Natale-Todeschi 2001.07005,
Cancès-Gallouët-Todeschi 1907.08305, Wu 2608.30693, Lascu 2411.15067 ; lot B :
Han 2211.07011, Matthes-Plazotta, Malamut-Sylvestre, Carrillo-Matthes-Wolfram
2003.03803). 14 / 471 = **3,0 %** (< 5 %). Deux lues (Blanes, 2105.09415),
douze restent en titre ou résumé, dont cinq prioritaires (Matthes-Plazotta,
Han, Legendre-Turinici, Carrillo et al. 2018, Malamut-Sylvestre).

**Ce que le tour a changé dans la note (v0.2).**

1. **Prédiction « Strang avec sous-pas JKO reste d'ordre 1 » : ni démontrée ni
   contredite dans la littérature trouvée.** Aucun résultat des canaux du
   tour 2 (articles, thèses, dépôts) ne combine Strang ou Trotter avec JKO ou
   Fisher-Rao. Trois éléments appuient la prédiction, de natures différentes :
   théorie classique de la composition (Blanes §1.3 : la composition d'une
   méthode et de son adjoint est symétrique, donc d'ordre pair ; l'adjoint de
   l'Euler implicite est l'Euler explicite), mécanisme de l'énergie modifiée
   (Halmos-Hanin Th. 2, sous les hypothèses de régularité A1-A5 : chaque pas JKO
   suit un flot d'énergie J − (η/4)\|∂J\|² à O(η²), d'où un biais O(η) par
   sous-pas) et le dépôt qui écrit « Lie-Trotter, Strang non testé ». L'argument
   de l'énergie modifiée est **une inférence de cette note**, pas un énoncé de
   l'article, qui ne traite aucun splitting.
2. **Carlier et al. Th. 3.16 : la condition ε\|log ε\| = O(τ²) est suffisante,
   pas nécessaire** (Baradat et al., Rem. 2.10, sur le tore) ; la nuance du §2bis
   est mise à jour, la conversion de convention de ε reste à vérifier.
3. Lignes 2, 5, 7 (seconde moitié) du §3 recoupées ; ligne 10 scindée ; ligne 4
   renforcée par Blanes ; ligne 6 précisée (O(τ) sous régularité, O(√τ) sous
   λ-convexité seule).
4. Nouvelle ligne 12 (aucun splitting avec JKO publié ne donne d'ordre) et
   nouvelle ligne 13 (outil de référence gaussien, licence absente).

**Prochain tour (canaux différents).** Version revue de pmRC8xTiym à sa
publication (décisions ICLR 2027) ou demande à ses auteurs ; thèse de Lin
(dépôt) ; lecture des cinq entrées prioritaires ci-dessus ; revue de
Crucinio-Pathiraja si une version publiée apparaît ; vérification de la
conversion de ε entre Carlier (`W_ε² = ∫c + εH(γ)`) et Baradat et al.
(éq. 5, `ε/2 Δρ`) sur un cas gaussien en forme close.
