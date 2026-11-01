# Note de lecture — NH-20, Lie O(τ) contre Strang O(τ²) dans un splitting W₂/Fisher-Rao (v0, 2026-11-01)

Statut : v0, tour de veille 1. Aucune préreg n'est gelée ni modifiée ; aucun
calcul n'a été lancé pour cette note. Le contradicteur n'est pas obligatoire
pour ce chantier (spec du verrou, §5.1). Le premier essai (campagne Lie/Strang
du 30/10, « M12 ») est déclaré au §4 : ordre local non identifiable, exposants
médians de S à −0,72 et −0,03 (`docs/notes/2026-11-01-jko-splitting-results.md:47-52`).

## 1. Question que la campagne prétend trancher (une phrase)

Dans un schéma qui découpe un flot de gradient en un sous-pas de transport W₂
et un sous-pas de réaction Fisher-Rao (ou, comme en M12, en deux sous-pas W₂
d'énergies partielles), la composition palindromique de Strang atteint-elle
l'ordre 2 en temps quand celle de Lie reste d'ordre 1, et sous quelles
conditions (nature des sous-pas, régularisation entropique ε, grille) cet ordre
est-il mesurable ?

Rappel de portée : la préreg M12 dit elle-même qu'elle est « related to NH-20,
not a direct test of it » ; ses deux sous-pas sont des pas JKO entropiques
d'énergies partielles, pas un pas W₂ suivi d'un pas FR
(`docs/prereg/2026-10-30-jko-splitting-lie-strang.md:36-40`).

## 2. Sources primaires lues en entier

Texte intégral téléchargé depuis arXiv le 01/11 et lu (détail des pages au §7).
Identifiants résolus par l'API arXiv (titre, auteurs, date) et, pour les DOI,
par Crossref (`works/<doi>`).

| id résolu | auteurs, année | pages lues | ce qu'elle établit (pour NH-20) |
|---|---|---|---|
| arXiv:1602.04457 (= DOI 10.1137/16m106666x, SIAM J. Math. Anal. 49:1100-1130) | Gallouët, Monsaingeon 2016 (revu 2017) | 1-30 | Schéma « JKO splitting » : un pas JKO de Monge-Kantorovich (W₂, masse conservée) puis un pas JKO de Fisher-Rao, même énergie F dans les deux (éq. 4.4-4.5, p. 13) ; les auteurs le lisent comme un splitting d'opérateurs (p. 13). Convergence vers une solution faible sous hypothèses de compacité forte (Th. 4, p. 17-18), vérifiées pour les énergies internes sous (H) (§5, p. 20-29). **Aucun ordre ni taux**, pas de version symétrique, pas de numérique. (H) exclut l'entropie de Boltzmann, non convexe pour FR (p. 21). |
| arXiv:1704.04541 (= DOI 10.1051/cocv/2018001, ESAIM COCV 25:8, 2019) | Gallouët, Laborde, Monsaingeon 2017 | 1-27 | Même alternance W₂ puis FR avec deux fonctionnelles distinctes (éq. 3.4) ; existence de solutions faibles pour réaction-diffusion-advection, systèmes proies-prédateurs, Hele-Shaw (Th. 3.2, 4.2, 5.1) ; numérique ALG2-JKO + Newton pointwise pour FR (§4). **Aucun ordre ni taux.** C'est le « Laborde 2017 WFR/JKO splitting » du brief. |
| arXiv:1506.00126 (= DOI 10.1515/9783110430417-013, chapitre De Gruyter, 2017) | Laborde 2015 (publié 2017) | 1-21 | Systèmes multi-espèces avec interactions croisées non proportionnelles : pas des flots de gradient ; schéma JKO **semi-implicite** (mesure gelée dans Vᵢ[ρ]), existence et unicité (Th. 2.3). Pas de splitting au sens Lie/Strang, pas d'ordre. C'est le « Laborde 2017 » de la revue de fond (cadre de J^asym). |
| arXiv:1512.02783 (= DOI 10.1137/15m1050264, SIAM J. Math. Anal. 49:1385-1418) | Carlier, Duval, Peyré, Schmitzer 2015 (revu 2017) | 1-28 (preuves de Γ-convergence §2.3-2.4 et estimations a priori §3.4-3.5 parcourues, énoncés lus) | Γ-convergence de l'OT entropique (Th. 2.7) ; convergence du JKO entropique vers l'EDP si ε\|log ε\| = O(τ²) (Th. 3.16, p. 17) ; ε = o(τ) nécessaire (p. 10) ; §4 : sur grille fixe, pour τ et ε très petits, le schéma « gèle » (p. 21-22). Énoncé recopié et vérifié au §2bis. |
| arXiv:2501.14660 (= DOI 10.1007/s11005-026-02065-9, Lett. Math. Phys. 116:42) | Hernandez, Pastorello, De Palma 2025 (revu 2026) | 1-23 | Propagation du chaos pour un MoE entraîné par flot de gradient continu : E W₂²(μᴺ, μ_t) ≤ C N^(−2/d), d > 4 (Th. 1.1). **Temps continu** : ni JKO, ni discrétisation, ni splitting. Ne concerne NH-20 que par l'image « experts = particules ». |
| arXiv:2605.31369 | Caucheteux, Bonet, Korba 2026 (ICML, PMLR 306) | 1-17 + annexes A, C, G parcourues (56 p.) | Cadre « Generative Wasserstein Flows » : méthodes génératives comme JKO paramétriques ; équivalences S-JKO = VWGF (Prop. 3.1), descente sur l'enveloppe de Moreau = JKO pour ε = 2τ (Prop. 3.2) ; JKO paramétrique = discrétisation implicite **d'ordre 1** d'un flot préconditionné (Prop. 6.3). Splitting cité en une ligne (Wibisono, Salim), aucun ordre Lie/Strang. |
| arXiv:2002.03035 | Salim, Korba, Luise 2020 (NeurIPS) | 1-16 | Schéma forward-backward dans W₂ (pas explicite sur E_F, JKO sur H, éq. 17-18) ; taux d'**optimisation** O(1/n) et linéaire (Cor. 10-11) sous convexité géodésique ; non biaisé, contrairement au forward-flow (ULA). Pas d'ordre de discrétisation du flot. |
| arXiv:2511.18060 (v3, 16/09/2026) | Crucinio, Pathiraja 2025 | 1-13 + annexes A-B parcourues | **Antériorité directe la plus proche** : splitting de Lie du flot WFR (W puis FR, ou FR puis W) avec sous-flots **exacts** ; erreur O(γ) (rappel, p. 3) ; EDP perturbée explicite (Prop. 2.2-2.3) ; l'ordre des opérateurs change la vitesse de convergence, cas gaussien en forme close (§3). **Strang n'est pas traité** ; schémas numériques des sous-flots laissés en travail futur (p. 13). |
| arXiv:2109.02792 | Liu, Wang, Wang 2021 | 1-23 | Splitting de **Strang d'ordre 2** pour réaction-diffusion à bilan détaillé où réaction et diffusion dissipent la même énergie libre (formulation énergétique variationnelle) ; sous-pas **d'ordre 2** (Crank-Nicolson, intégrateur exact) ; ordre 2 constaté par test de Cauchy avec Δt = h (§4). Diffusion eulérienne, pas un pas JKO W₂. |
| arXiv:1802.08089 | Wibisono 2018 (COLT, PMLR 75) | 1-20 + annexe B | Langevin comme optimisation composite dans W₂ ; une méthode symétrique (composée avec son adjoint) est d'ordre pair ≥ 2 ; avant/arrière (JKO) sont adjoints d'ordre 1 (annexe B.3-B.4) ; algorithme symétrisé SLA, consistant pour OU gaussien (Ex. 3). Formes closes gaussiennes. |
| arXiv:1005.0998 | Clément, Maas 2010 | 1-20 (preuves §2 parcourues) | Formule de Trotter pour flots de gradient métriques : J²∘J¹ itérés convergent vers la solution EVI (Th. 1.1), sous λ-convexité de chaque partie ; reste vrai avec les semi-groupes exacts (Rem. 1.6). **Convergence sans taux.** |
| arXiv:2410.03367 | Cancès, Monsaingeon, Natale 2024 | 1-10, 30-36 (preuves §3-4 parcourues) | Le JKO n'est que d'ordre 1 en temps (p. 2) ; schéma volumes finis point milieu non linéaire, ordre 2 constaté en temps et en espace avec deux raffinements de maillage (§5). Référence d'ordre 2 possible. |

Lus en résumé seulement (abstract arXiv, marqués **[résumé moteur]** partout où
ils sont cités) : Halmos, Hanin, arXiv:2511.14827 ; Baradat, Cherf,
arXiv:2604.08283 ; Chu, Jacobs, arXiv:2511.10884 ; Gallouët, Natale, Todeschi,
arXiv:2209.14622 ; Adams, Duong, dos Reis, arXiv:2105.11146 ; Bou-Rabee,
arXiv:2608.25279 ; Carlier, Laborde, arXiv:1606.04793 ; Bernton,
arXiv:1802.08671 ; Hraivoronska, Santambrogio, DOI 10.1093/imanum/drag053
(titre seul) ; Strang 1968, DOI 10.1137/0705041 (métadonnées seules).

### 2bis. Carlier et al. 2017, théorème 3.16 : énoncé, hypothèses, vérification

**Énoncé recopié (arXiv v2, p. 17).** « Theorem 3.16. In the above setup
(Assumption 3.1 and (3.57)), the limit curve of measures P solves the evolution
equation ∂ₜP = Δp(P) + div(P∇v), P|ₜ₌₀ = ρ₀. »

**« Setup » (p. 9-10 et 17), recopié en substance.**

- Schéma (3.12) : ρ^(ε,τ,k+1) ∈ argmin_ρ J_{ε,τ}(ρ^(ε,τ,k), ρ), avec
  J_{ε,τ}(μ, ν) = W_ε²(μ, ν)/(2τ) + F(ν) (3.3) et
  W_ε²(μ, ν) = inf_{γ∈Π(μ,ν)} (c, γ) + ε H(γ) (2.5), c(x, y) = |x − y|²,
  H l'entropie **relative à la mesure de Lebesgue sur ℝⁿ × ℝⁿ** (2.1).
- (3.57) : deux suites εₖ, τₖ → 0 avec −εₖ log εₖ ≤ C τₖ² pour une constante
  C ∈ (0, ∞) fixée ; Nₖ τₖ → T (3.58) ; Pₖ = P^(εₖ,τₖ,Nₖ) interpolation
  constante par morceaux (3.13) ; P limite (sous-suite) forte dans Lᵐ avec
  p(Pₖ) → p(P) dans L¹. Une étape (Prop. 3.7, p. 12) demande en plus
  0 < ε ≤ ε\|log ε\| ≤ C₁τ² (3.26), donc ε ≤ 1/e.
- Assumption 3.1 (p. 9-10) : F = V + U sur P₂ʳ(ℝⁿ) (mesures à densité,
  second moment fini) ; V(ν) = ∫ v ν avec **v ≥ 0 et lipschitzien** ;
  U(ν) = ∫ u(ν(x)) dx avec u convexe, C² sur (0, ∞), u(0) = 0, superlinéaire
  (3.6) ; u(s) ≥ −C s^α avec n/(n+2) < α < 1 (3.7) ; pression
  p(s) = s u′(s) − u(s) avec p(s) ≤ C sᵐ, p′(s) ≥ s^(m−1)/C (3.8) ; borne
  Lᵐ (3.9). Donnée initiale ρ₀ ∈ P₂ʳ(ℝⁿ), F(ρ₀) < ∞ (et H(ρ₀) < ∞ pour la
  Prop. 3.4).
- Remarques des auteurs : la condition ε\|log ε\| = O(τ²) n'est pas connue
  pour optimale, mais ε = o(τ) est nécessaire (p. 10). Fig. 4.3 (p. 22) trace
  la frontière ε = C τ² \|log τ\| avec C = 10³.

**Vérification aux réglages du labo.** Faits (calculés ici, Python, à partir
des valeurs du code) :

| réglage | où dans le code | ε | τ = h | ε\|log ε\| / τ² | ε / τ | Δx² / ε (grille 16 sur [−2, 2]) |
|---|---|---|---|---|---|---|
| article, « rigorous path » historique | `paper/main.tex:345` ; `kiki_flow_core/track2_paper/paper_run.py:86-88` ; `kiki_flow_core/track2_paper/full_jko_solver.py:12` | 0,01 | 0,05 | 18,4 | 0,2 | — (grille de l'article différente) |
| défaut ignoré avant ERRATUM §9 | `ERRATUM.md:271` (§9) ; `kiki_flow_core/wasserstein_ops.py:82` | 0,05 | 0,05 | 59,9 | 1 | — |
| JKO entropique actuel, M5 et M12 | `kiki_flow_core/master_equation.py:297-298` (reg = h²) ; `kiki_flow_core/track2_paper/jko_splitting.py:17-18` (sous-pas s, reg = s²) ; `experiments/implicit_jko_certified_validation_2026_10_27.py:75,80` | h² | 0,1 / 0,05 / 0,025 / 0,0125 | 4,6 / 6,0 / 7,4 / 8,8 | h | 7 / 28 / 114 / 455 |

Grille : `experiments/jko_splitting_2026_10_30.py:75` (GRID = 16) et
`experiments/implicit_jko_certified_validation_2026_10_27.py:181-182`
(linspace(−2, 2)), donc Δx = 4/15 ; échelle de pas :
`experiments/jko_splitting_2026_10_30.py:81`.

Lecture (inférences, séparées des faits) :

1. (3.57) porte sur des **suites** avec une constante C fixée ; une paire
   isolée (ε, h) ne la viole pas au sens strict. Ce qu'on peut vérifier est la
   loi ε(h) que suit le code quand h décroît.
2. ε = 0,01 fixe (article) : ε\|log ε\|/τ² → ∞ quand τ → 0, et ε = o(τ) est
   faux. La loi ε = ατ que cite l'article (`paper/main.tex:119`) donne
   ε\|log ε\|/τ² ~ α\|log τ\|/τ → ∞. Les deux sortent de (3.57) et de la
   condition nécessaire ε = o(τ). Accord avec `revue-de-fond-2026-11-01.md:699`
   (facteur ≈ 18 à ε = 0,01, h = 0,05).
3. ε = h² (code actuel) : ε = o(τ) est vrai, mais ε\|log ε\|/τ² = 2\|log h\|
   croît sans borne (4,6 → 8,8 sur l'échelle M12) ; (3.57) échoue donc aussi
   asymptotiquement, d'un facteur logarithmique seulement. **Nuance au rapport
   du 01/11** : le constat « non tenu à ε = 0,01, h = 0,05 »
   (`analyse-profonde/synthese/rapport.md:258` ; spec §1) vaut pour le réglage
   de l'article ; M5 et M12 tournaient à ε = h², où l'écart est logarithmique.
4. Les autres hypothèses ne sont pas remplies non plus, indépendamment de ε :
   le théorème est posé sur ℝⁿ continu, une seule densité, F = V + U local ;
   le labo travaille sur une grille de 16 points (discrétisation du §4 de
   Carlier, que le théorème ne couvre pas), avec plusieurs espèces, un
   couplage croisé J_sym⟨ρᵢ, ρₖ⟩ et un terme de Turing lissé non convexe
   (préreg M12, `docs/prereg/2026-10-30-jko-splitting-lie-strang.md:70-74`),
   des potentiels linéaires α(x ± 1,5) non positifs sur ℝ. Aucun résultat de
   convergence de Carlier ne s'applique donc au JKO du labo ; a fortiori aucun
   résultat d'ordre.
5. Le « gel » sur grille fixe que Carlier décrit au §4 (p. 21-22 : « the
   transport cost to the next grid point can be prohibitive and the gradient
   flow scheme 'freezes' ») est le même phénomène que le « grid pinning » du
   labo (`docs/notes/2026-10-25-jko-step-identification.md`, cité dans la
   préreg M12 l. 101-111) ; les rapports Δx²/ε de 7 à 455 (1 820 pour les
   demi-pas A à h = 0,0125, préreg l. 107-108) placent M12 dans ce régime.

## 3. Déjà publié

Chaque ligne : affirmation → sources (nature) ; « à recouper » si une seule
source ou deux sources de même nature.

1. Alterner un pas JKO W₂ et un pas JKO Fisher-Rao de la même énergie converge
   vers une solution faible de l'équation de réaction-advection-diffusion du
   flot WFR (KFR) → Gallouët-Monsaingeon arXiv:1602.04457 (preprint lu) +
   DOI 10.1137/16m106666x (version revue, métadonnées) ; étendu à deux
   fonctionnelles par arXiv:1704.04541. **Recoupé** (preprint + version revue).
2. Un splitting de Lie de flots de gradient métriques converge (formule de
   Trotter), sans taux → Clément-Maas arXiv:1005.0998 + Gallouët-Monsaingeon
   Th. 4. **À recouper** (deux articles, même nature ; aucun code ni thèse lu).
3. Le splitting de Lie de WFR avec sous-flots exacts a une erreur O(γ) décrite
   par une EDP perturbée explicite, et l'ordre W-FR / FR-W change la vitesse de
   convergence → Crucinio-Pathiraja arXiv:2511.18060. **À recouper** (un seul
   preprint).
4. Avec des sous-flots exacts (ou d'ordre ≥ 2, ou appariés à leur adjoint),
   la composition symétrique est d'ordre 2 ; avec des sous-pas d'Euler
   implicite (dont le JKO est l'analogue), Strang est d'ordre 1 générique →
   Wibisono arXiv:1802.08089 annexe B (article) + tests du labo
   `tests/track2_paper/test_jko_splitting.py:117,130` (code, modèle linéaire)
   + Strang 1968, DOI 10.1137/0705041 (métadonnées seules). **Recoupé**
   (article + code), en dimension finie et en régime lisse seulement.
5. Un splitting de Strang d'ordre 2 existe pour réaction-diffusion à énergie
   commune, à condition que chaque sous-pas soit d'ordre 2 → Liu-Wang-Wang
   arXiv:2109.02792 (preprint, numérique incluse). **À recouper** (une source ;
   pas de version revue ni de code consulté).
6. Le JKO est d'ordre 1 en temps → Cancès-Monsaingeon-Natale arXiv:2410.03367
   p. 2 + Caucheteux-Bonet-Korba Prop. 6.3 (cas paramétrique) + Halmos-Hanin
   arXiv:2511.14827 **[résumé moteur]** (écart O(η²) au flot d'une énergie
   modifiée). **Recoupé** pour l'ordre 1 (deux articles lus), l'énergie
   modifiée reste **à recouper**.
7. JKO entropique : convergence vers le flot W₂ si ε\|log ε\| = O(τ²),
   ε = o(τ) nécessaire → Carlier et al. arXiv:1512.02783 (lu) + DOI
   10.1137/15m1050264 (version revue, métadonnées). **Recoupé.** Avec ε = ατ,
   diffusion supplémentaire (α/2)Δρ à la limite → article cité par
   `paper/main.tex:119-123` et Baradat-Cherf arXiv:2604.08283 **[résumé
   moteur]**. **À recouper.**
8. Sur grille fixe, pas petit et ε petit, le JKO entropique gèle → Carlier §4
   p. 21-22 (article) + note du labo `docs/notes/2026-10-25-jko-step-identification.md`
   et données M12 (`docs/notes/2026-11-01-jko-splitting-results.md:47-52`).
   **Recoupé** (article + données du labo).
9. Forward-backward (pas explicite sur le potentiel, JKO sur l'entropie) :
   non biaisé, taux d'optimisation de l'algorithme du gradient proximal →
   Salim-Korba-Luise arXiv:2002.03035 + Wibisono arXiv:1802.08089 (FB
   consistant pour OU, Ex. 8). **Recoupé** (deux articles lus, cadres distincts).
10. Des schémas en temps d'ordre 2 pour les flots W₂ existent (point milieu,
    BDF2, accéléré) → Cancès-Monsaingeon-Natale (lu) + Gallouët-Natale-Todeschi
    arXiv:2209.14622 et Chu-Jacobs arXiv:2511.10884 **[résumé moteur]**.
    **À recouper** pour BDF2 et accéléré.
11. Experts d'un mélange = particules d'un flot de Wasserstein (temps continu)
    → Hernandez et al. arXiv:2501.14660 (preprint lu + version revue DOI
    10.1007/s11005-026-02065-9, métadonnées). **Recoupé**, hors du cœur de
    NH-20.

## 4. Reste ouvert

**Question restante (formulation v0).** Mesurer l'ordre observé de Lie et de
Strang dans un splitting W₂/Fisher-Rao (ou W₂/W₂ d'énergies partielles) dont
les sous-pas sont des **mouvements minimisants JKO** (entropiques ou non), et
établir sous quelles conditions sur (h, ε, Δx, tolérance des sous-pas) cet
ordre est identifiable ; en option, vérifier si une composition symétrique qui
apparie un demi-pas JKO à son adjoint explicite restaure l'ordre 2.

Ce que la littérature lue en dit (faits) : 1602.04457 et 1704.04541 prouvent
la convergence sans ordre ; 2511.18060 traite Lie avec sous-flots exacts,
pas Strang ; 2109.02792 traite Strang avec sous-pas d'ordre 2 non JKO ;
1802.08089 traite la symétrisation pour Langevin (W₂ seul, pas FR).

Preuve de l'absence (tour 1, §7 et `veille-journal.md`) : 8 requêtes arXiv
(33 identifiants distincts), 3 requêtes Crossref (60 notices), 4 requêtes
GitHub (3 dépôts), remontée de citations Semantic Scholar sur 7 sources
(615 lignes, 496 identifiants distincts, `snowball-tour1.tsv`) : aucun titre ne
combine Strang (ou « symmetric », « second order ») avec JKO ou Fisher-Rao,
hormis Bou-Rabee arXiv:2608.25279 (Strang de Langevin cinétique, autre objet,
**[résumé moteur]**) et Liu-Wang-Wang (sous-pas non JKO). Absence établie par
titres et résumés, pas par lecture : **à recouper** au tour 2 (OpenReview,
HAL, thèses Laborde et Caillet).

Inférences (séparées des faits) :

- La réponse attendue est en grande partie **prévisible** : avec des sous-pas
  JKO d'ordre 1, Strang devrait être d'ordre 1 comme Lie (ligne 4 du §3) ; avec
  des sous-flots exacts, Strang devrait être d'ordre 2 en régime lisse. Ce qui
  reste non trivial : (a) l'effet de ε(h) et de la grille sur l'ordre observé,
  (b) l'appariement adjoint dans W₂/FR, (c) le comportement hors régime lisse
  (énergie non convexe du labo). La contribution neuve est donc une **note
  numérique calibrée**, pas une découverte ; la spec la décrit déjà comme
  « seule contribution neuve de la ligne » (spec §6), à prendre dans ce sens
  étroit.
- M12 ne pouvait pas trancher : il viole les conditions C2 à C5 ci-dessous
  (ε = s², Δx²/ε de 7 à 1 820, plancher 1e-5 au-dessus de h³ à h = 0,0125,
  sous-pas JKO d'ordre 1). Son « non identifiable » est le résultat attendu par
  construction, comme sa préreg l'envisageait (l. 299-300).

**Condition explicite d'identifiabilité (v0).** Sur une échelle hₖ = h₀2⁻ᵏ,
k = 0..K (K ≥ 2), mesurer l'exposant local p + 1 du défaut d'un schéma Y
(auto-convergence s_Y(h) = ‖Y_h − Y_{h/2}∘Y_{h/2}‖ ou erreur à une référence)
exige simultanément :

- **C1, régime asymptotique** : sur toute l'échelle, le terme dominant
  c_{p+1}h^{p+1} l'emporte sur le suivant, |c_{p+2}|h₀ ≤ 0,1 |c_{p+1}| ; se
  constate par deux exposants adjacents dans la même bande (convention M12).
- **C2, régularisation entropique** : (inférence, à vérifier par le dessin P1)
  à F = 0 le pas JKO entropique floute avec une variance ε/2 par pas (plan
  ∝ exp(−|x − y|²/ε), cf. préreg M12 l. 321-323). Avec ε = s², un pas h
  floute de h²/2 et deux demi-pas de h²/4 : l'écart ∝ h² entre dans s_Y avec
  l'exposant 2, quel que soit le schéma, et masque l'exposant 3 de Strang.
  Avec ε = αs, les deux variances valent αh/2 : le flou est cohérent d'un pas
  à l'autre, l'ordre se mesure par rapport au flot modifié (diffusion
  supplémentaire d'ordre α), et α → 0 se traite à part. Condition : soit
  ε = αs avec référence au même α, soit ε ≤ 0,1 c_{p+1} h_min^{p+1} (ordre de
  grandeur ; pour Strang, ε ≪ h³).
- **C3, grille** : pas de gel ni de flou non résolu,
  Δx ≤ 0,1 min(√ε, |v| h_min) où v est la vitesse du flot ; pour ε = s² cela
  donne Δx ≤ h_min/10, soit N ≥ 3 200 points sur [−2, 2] pour h_min = 0,0125
  (M12 : N = 16).
- **C4, convergence des sous-pas** : avec m sous-pas par pas (3, 6, 9 pour S,
  Lie, Strang en auto-convergence, préreg l. 233-235) et une tolérance δ_S sur
  la distance à la solution exacte du sous-problème (pas seulement sur le
  résidu de point fixe), m δ_S ≤ 0,1 c_{p+1} h_min^{p+1} ; pour Strang avec
  c ≈ 1 et h_min = 0,0125, δ_S ≲ 2e-8, contre un plancher heuristique 1e-5 en
  M12 (préreg l. 230-235).
- **C5, nature des sous-pas** : l'ordre 2 de Strang n'est attendu qu'avec des
  sous-flots exacts, d'ordre ≥ 2 ou appariés à leur adjoint ; avec des
  sous-pas JKO purs, la question « Lie contre Strang » ne discrimine rien
  (ordre 1 attendu des deux côtés).
- **C6, régularité** : densités lisses et strictement positives, énergie
  λ-convexe pour chaque partie, minimiseur unique de chaque sous-problème (pas
  de changement de branche, contrairement au B non convexe de M12, préreg
  l. 373-376).

## 5. Dessin minimal qui tranche la question ouverte

**Bras.** Problème 1D WFR gaussien, F = KL(ρ‖N(m_π, σ_π²)), découpé en flot W₂
et flot FR : (i) Lie W-FR et FR-W, (ii) Strang W_{h/2}∘FR_h∘W_{h/2} et
FR_{h/2}∘W_h∘FR_{h/2}, chacun avec trois natures de sous-pas : exacts (formes
closes, 2511.18060 §3 et annexe B), JKO résolus sur grille fine sans
régularisation (W₂ en 1D par fonctions quantiles, FR par Newton point par
point comme 1704.04541 §4), et symétrique-adjoint (demi-pas explicite +
demi-pas JKO). Référence : solution exacte du flot WFR gaussien (2511.18060,
éq. 3.1-3.2).

**Nul calibré.** Pentes prédites avant tout calcul : erreur globale à T fixé
d'ordre 1 pour Lie (toutes natures), 2 pour Strang exact et
symétrique-adjoint, 1 pour Strang à sous-pas JKO. Calibration du pipeline sur
le cas exact, où la prédiction est un théorème classique : si Strang exact
n'atteint pas une pente ≥ 1,8, le montage est défaillant et rien d'autre n'est
lu.

**Critère.** Pente des moindres carrés de log(erreur) sur six niveaux de h, avec
deux exposants adjacents dans la même bande (C1) ; C2 à C4 vérifiés et
consignés avant lecture (ε = 0, N ≥ 4 096, tolérance des sous-problèmes
≤ 1e-10).

**Coût.** Inférence : formes closes en secondes ; version grille, 1D,
N = 4 096, 6 niveaux, ≈ 10³ pas au plus fin : ≲ 2 h CPU sur electron-server
en `nice 19`, sans GPU. Heures humaines : ≈ 6 (code + tests) + 2 (lecture).
Aucun lancement avant le passage du verrou.

## 6. Trois tests du verrou (état à ce tour)

- **Saturation : non atteinte.** Le tour 1 a apporté cinq entrées qui changent
  la note (2511.18060, 2109.02792, 1802.08089, 2511.14827, 2604.08283). Il faut
  au moins un tour complet sans nouvelle entrée retenue ; prochain tour :
  citations amont/aval de 2511.18060, 2109.02792 et 1802.08089.
- **Recoupement : partiel.** Lignes 2, 3, 5, 7 (seconde moitié) et 10 du §3,
  ainsi que la preuve d'absence du §4, sont « à recouper ». La condition C2 est
  une inférence non testée.
- **Couverture : 4 canaux sur 6.** Interrogés : arXiv, Crossref-journaux,
  thèses (par les DOI ABES de Crossref seulement, pas de HAL direct), code
  (GitHub). Restent : OpenReview, bioRxiv (pour la forme ; faible pertinence
  attendue), HAL en direct, Zenodo et Hugging Face pour les données.

## 7. Journal de lecture

| date | source | pages ou sections |
|---|---|---|
| 2026-11-01 | lab : `docs/notes/2026-11-01-jko-splitting-results.md` | entier (68 lignes) |
| 2026-11-01 | lab : `docs/prereg/2026-10-30-jko-splitting-lie-strang.md` | entier (450 lignes) |
| 2026-11-01 | lab : `docs/notes/2026-10-25-jko-step-identification.md` | §0-3, 5-6 |
| 2026-11-01 | lab : `paper/main.tex` | l. 100-125, 200-212, 290-353, 470-480, 885-905 |
| 2026-11-01 | lab : `ERRATUM.md` | titres des §1-16, en-tête, §9 (ligne 271) |
| 2026-11-01 | analyse du 01/11 : `synthese/rapport.md` | l. 11, 27, 58-61, 95-100, 149, 152, 203, 235, 258, 266 |
| 2026-11-01 | veilles du 01/11 : `veille-2026-11-01.md` thèse 3 ; `revue-de-fond-2026-11-01.md` R3 | l. 107-138, 322, 338, 409-424 ; l. 169-219, 555-567, 624-629, 698-700 |
| 2026-11-01 | arXiv:1512.02783 Carlier et al. | p. 1-28 ; §3.1 et Th. 3.16 lus mot à mot ; §2.3-2.4, 3.4-3.5 parcourus |
| 2026-11-01 | arXiv:1602.04457 Gallouët-Monsaingeon | p. 1-30 ; §1-4 lus, §5 parcouru (énoncés lus) |
| 2026-11-01 | arXiv:1704.04541 Gallouët-Laborde-Monsaingeon | p. 1-27 ; §1-3 et §4 numérique lus, §5-6 parcourus |
| 2026-11-01 | arXiv:1506.00126 Laborde | p. 1-21 ; §1-3 lus, §4-7 parcourus |
| 2026-11-01 | arXiv:2501.14660 Hernandez et al. | p. 1-23 ; §1-2, 5 lus, §3-4 parcourus |
| 2026-11-01 | arXiv:2605.31369 Caucheteux et al. | p. 1-17, annexe A ; annexes C, G parcourues |
| 2026-11-01 | arXiv:2002.03035 Salim et al. | p. 1-16 (corps entier), annexes parcourues |
| 2026-11-01 | arXiv:2511.18060 Crucinio-Pathiraja | p. 1-13 ; annexes A-B parcourues |
| 2026-11-01 | arXiv:2109.02792 Liu-Wang-Wang | p. 1-23 ; §1, 3 (début), 4 lus |
| 2026-11-01 | arXiv:1802.08089 Wibisono | p. 1-8, annexe B |
| 2026-11-01 | arXiv:1005.0998 Clément-Maas | p. 1-4 (Th. 1.1, remarques), §2-3 parcourus |
| 2026-11-01 | arXiv:2410.03367 Cancès-Monsaingeon-Natale | p. 1-3, §5 (p. 31-36) |
