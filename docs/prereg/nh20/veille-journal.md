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
