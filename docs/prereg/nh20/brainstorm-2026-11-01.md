# Brainstorm NH-20 — rendre l'ordre Lie/Strang identifiable (2026-11-01)

**Aucun de ces dessins ne part avant que le verrou soit passé** (note de
lecture complète, trois tests ; pas de contradicteur obligatoire pour NH-20,
spec §5.1). Aucun n'est une préreg : ce sont des candidats à comparer.

## Question ouverte rappelée

D'après `reading-note.md` §4 : mesurer l'ordre observé de Lie et de Strang dans
un splitting W₂/Fisher-Rao (ou W₂/W₂ d'énergies partielles) dont les sous-pas
sont des mouvements minimisants JKO, et établir sous quelles conditions
(C1 régime, C2 régularisation ε(h), C3 grille, C4 convergence des sous-pas,
C5 nature des sous-pas, C6 régularité) cet ordre est identifiable ; en option,
tester l'appariement d'un demi-pas JKO avec son adjoint explicite.

Premier essai déclaré : M12, ordre non identifiable, exposants médians de S à
−0,72 et −0,03 (`docs/notes/2026-11-01-jko-splitting-results.md:47-52`) ; M12
viole C2 à C5 (note §4).

Prédictions écrites avant tout calcul (théorie classique, note §3 ligne 4) :

| sous-pas | Lie | Strang |
|---|---|---|
| flots exacts | 1 | 2 |
| JKO (Euler implicite) | 1 | 1 |
| demi-pas explicite + demi-pas JKO (adjoints) autour d'un sous-flot exact | — | 2 |

Ordres de l'erreur **globale** à T fixé ; l'exposant local d'auto-convergence
vaut ordre + 1.

## Dessins

### P1 — Gaussien W₂ seul (Ornstein-Uhlenbeck), formes closes

- **Bras.** F = KL(ρ‖N(0, σ²)) découpé en potentiel (A, transport pur, flot
  exact affine) et entropie (B, flot de la chaleur exact) ; Lie, Strang,
  Strang symétrique-adjoint ; sous-pas exacts, JKO (forme close gaussienne
  pour le JKO de l'entropie et du potentiel quadratique, Wibisono 2018 Ex. 8 ;
  Salim et al. 2020 §5), explicites. Bras entropique ε ∈ {αh, h²} seulement si
  une forme close de l'OT entropique entre gaussiennes est trouvée et résolue
  au tour 2 ; sinon renvoyé à P3.
- **Nul calibré.** Tableau de prédictions ci-dessus ; la solution exacte
  (moyenne et variance d'OU) est la référence, sans erreur d'espace.
- **Critère de réfutation.** Pour chaque bras, pente des moindres carrés de
  log(erreur W₂ à T = 1) sur 8 niveaux h = 2⁻² … 2⁻⁹, deux exposants adjacents
  dans la même bande ([0,75 ; 1,25] pour 1, [1,75 ; 2,25] pour 2). Une pente
  hors de la bande prédite réfute la prédiction de ce bras ; si le bras
  « Strang exact » sort de [1,75 ; 2,25], le montage est déclaré défaillant et
  rien d'autre n'est lu.
- **Coût.** < 1 min CPU (electron-server, `nice 19`) ; ≈ 3 h humaines.
- **Ce que ça tranche.** La chaîne de mesure et la prédiction C5 dans le cas
  le plus simple (W₂ seul). Ne dit rien de FR ni de la grille.
- **Antériorité la plus proche.** Wibisono arXiv:1802.08089 (formes closes
  OU, symétrisation SLA consistante) ; Halmos-Hanin arXiv:2511.14827 (JKO-Flow
  sur Bures-Wasserstein, **[résumé moteur]**). Faible nouveauté : calibration.

### P2 — Gaussien WFR (transport W₂ + réaction FR), formes closes, Strang ajouté

- **Bras.** Flot WFR de KL(ρ‖N(m_π, σ_π²)) en 1D ; sous-flots exacts W et FR
  (formes closes, Crucinio-Pathiraja arXiv:2511.18060 §3, annexe B) ; Lie W-FR,
  Lie FR-W, Strang W_{h/2}∘FR_h∘W_{h/2}, Strang FR_{h/2}∘W_h∘FR_{h/2} ;
  deux régimes (σ_π² > σ₀² et σ_π² < σ₀², comme leur Fig. 3.1).
- **Nul calibré.** Lie ordre 1, Strang ordre 2 ; référence = solution exacte
  du flot WFR gaussien (2511.18060, éq. 3.1-3.2).
- **Critère de réfutation.** Mêmes bandes que P1 ; « Strang exact d'ordre 2
  pour WFR » est réfuté si la pente sort de [1,75 ; 2,25] alors que C1 tient
  (exposants adjacents cohérents).
- **Coût.** < 1 min CPU ; ≈ 4 h humaines (reprise des formules de l'annexe B,
  tests).
- **Ce que ça tranche.** NH-20 dans sa forme d'origine (R11 H7 : W₂ contre FR)
  avec sous-flots exacts, ce que 2511.18060 n'a pas fait pour Strang. Réponse
  attendue par la théorie classique : le résultat vaut comme vérification
  publiée, pas comme découverte.
- **Antériorité la plus proche.** Crucinio-Pathiraja arXiv:2511.18060 (Lie
  seulement) ; Gallouët-Monsaingeon arXiv:1602.04457 (schéma, sans ordre).

### P3 — Richardson sur grille fine 1D, sous-pas JKO résolus à convergence

- **Bras.** 1D, N ∈ {1 024, 4 096} ; énergie non quadratique (potentiel à deux
  puits + entropie) avec réaction FR logistique en option ; sous-pas JKO W₂
  sans régularisation par fonctions quantiles (problème convexe, Newton à
  hessienne tridiagonale), pas FR par Newton point par point
  (Gallouët-Laborde-Monsaingeon arXiv:1704.04541 §4) ; Lie, Strang,
  Strang symétrique-adjoint ; puis bras entropiques avec Sinkhorn en log-domaine
  pour ε ∈ {0, αs, s²} (α ∈ {0,01 ; 0,1}).
- **Nul calibré.** Prédictions du tableau ; prédiction C2 écrite à l'avance :
  avec ε = s², l'exposant d'auto-convergence de tous les schémas plafonne à 2 ;
  avec ε = αs il suit la nature des sous-pas.
- **Critère de réfutation.** Ordre de Richardson p̂ = log₂((u_h − u_{h/2}) /
  (u_{h/2} − u_{h/4})) sur 6 niveaux, cohérent sur deux triplets adjacents ;
  C3 et C4 vérifiés et consignés avant lecture (Δx ≤ 0,1 min(√ε, |v|h_min),
  tolérance des sous-problèmes ≤ 1e-10 sur la distance au minimiseur, pas
  seulement le résidu). La prédiction C2 est réfutée si, à ε = s², un schéma
  dont les sous-pas sont d'ordre 2 montre un exposant d'auto-convergence
  ≥ 2,75.
- **Coût.** Bras non entropiques : ≲ 1 h CPU ; bras entropiques (noyau dense
  N², 10³ pas au niveau le plus fin) : 2 à 6 h CPU sur electron-server en
  `nice 19`, ou KXKM CPU en `nice 19` (pas de GPU). ≈ 8 h humaines.
- **Ce que ça tranche.** Les conditions C2-C4 de la note : quel ε(h) et quelle
  grille rendent l'ordre lisible ; la prédiction « Strang à sous-pas JKO =
  ordre 1 » hors du cas gaussien.
- **Antériorité la plus proche.** Liu-Wang-Wang arXiv:2109.02792 (test de
  Cauchy avec Δt = h, Strang à sous-pas d'ordre 2) ; Cancès-Monsaingeon-Natale
  arXiv:2410.03367 §5 (raffinements combinés) ; Carlier et al. arXiv:1512.02783
  §4, Fig. 4.2-4.3 (cartes d'erreur ε-τ).

### P4 — Solution manufacturée non gaussienne (potentiel dépendant du temps)

- **Bras.** On choisit ρ*(t, x) lisse et positive sur [−2, 2] (par exemple
  mélange de deux bosses qui se déplacent) ; en 1D, le flux est
  j = −∫ ∂ₜρ* et un potentiel V(t, x) tel que ρ* résout ∂ₜρ = ∇·(ρ∇(V + log ρ))
  s'obtient par ∇V = −j/ρ* − ∇log ρ* ; même construction pour un taux de
  réaction FR. Schémas Lie / Strang / symétrique-adjoint, V évalué au début du
  pas (Lie) ou au milieu (Strang), sous-pas comme P3.
- **Nul calibré.** Prédictions du tableau, avec une erreur supplémentaire
  d'ordre 1 si V(t) n'est pas évalué au point milieu pour Strang (à vérifier
  sur le bras « V figé au début du pas », prédit ordre 1).
- **Critère de réfutation.** Pente de l'erreur L¹ et W₂ à ρ*(T) sur 6 niveaux ;
  bandes de P1.
- **Coût.** ≲ 2 h CPU (electron-server, `nice 19`) ; ≈ 6 h humaines.
- **Ce que ça tranche.** L'ordre sur une densité non gaussienne, avec une
  référence exacte qui ne dépend ni de la grille ni d'une extrapolation ; isole
  l'erreur de splitting de l'erreur de référence.
- **Antériorité la plus proche.** Méthode des solutions manufacturées
  (classique, non référencée ici, à résoudre si citée) ; aucune trouvée pour un
  splitting JKO W₂/FR au tour 1.

### P5 — Décomposition des erreurs sur le problème T2 du labo

- **Bras.** Deux à trois cellules convexes de M12 (γ = 0, α ∈ {2, 5}), partie A
  seule puis A + B ; facteurs N ∈ {16, 64, 256} × loi ε ∈ {s², αs} ×
  tolérance des sous-pas ∈ {1e-6, 1e-9} ; mesures séparées de l'erreur de
  splitting (X_h contre S_h au même JKO), de l'erreur de temps (S_h contre
  S_{h/2}∘S_{h/2}), du biais entropique (même h, ε → 0) et du gel (même h, N
  croissant).
- **Nul calibré.** Chaque terme suit la loi d'échelle prédite (C2 : ∝ h² pour
  ε = s² ; C3 : décroît avec N ; C4 : constant en h) ; M12 est expliqué si le
  terme de gel domine aux rapports Δx²/ε de 7 à 455.
- **Critère de réfutation.** L'explication « M12 dominé par le gel » est
  réfutée si, à N = 256 et ε = αs, l'exposant de S reste hors de [1,5 ; 2,5)
  (bande de la préreg M12) sur la majorité des cellules.
- **Coût.** Le solveur certifié sur noyau dense N² monte vite : estimation
  6 à 30 h CPU sur KXKM CPU en `nice 19`. Au-delà de 6 h CPU, c'est une
  campagne lourde au sens du verrou (spec §4.3) : préreg L1 et verrou complet
  obligatoires. ≈ 8 h humaines.
- **Ce que ça tranche.** Pourquoi M12 n'a pas pu identifier l'ordre
  (rétrospectif) ; ne dit rien de l'ordre de Strang en régime propre.
- **Antériorité la plus proche.** Note du labo
  `docs/notes/2026-10-25-jko-step-identification.md` (gel de grille) ; Carlier
  et al. §4 ; Hraivoronska-Santambrogio DOI 10.1093/imanum/drag053 (JKO
  entièrement discret, titre seul).

### P6 — Prédiction par l'équation modifiée du JKO

- **Bras.** À partir de l'énergie modifiée J^η = J − (η/4)∫‖∇δJ/δρ‖²ρ
  (Halmos-Hanin arXiv:2511.14827, **[résumé moteur]**, à lire en entier avant
  usage), calculer le coefficient du défaut local de Strang à sous-pas JKO et
  le comparer aux défauts mesurés en P1 et P3.
- **Nul calibré.** Coefficient prédit ; égalité attendue à 10 % près en régime
  asymptotique.
- **Critère de réfutation.** Écart > 25 % sur trois niveaux de h où C1-C4
  tiennent.
- **Coût.** Calcul symbolique + réutilisation des sorties P1/P3 : < 1 h CPU ;
  ≈ 6 h humaines (dont la lecture).
- **Ce que ça tranche.** Si la théorie du JKO à l'ordre 2 rend compte
  quantitativement du défaut de Strang ; donnerait une prédiction chiffrée
  pour une préreg.
- **Antériorité la plus proche.** Halmos-Hanin (énergie modifiée du JKO) ;
  Crucinio-Pathiraja (EDP perturbée du splitting de Lie exact).

## Classement « ce que ça tranche / coût »

| rang | dessin | ce que ça tranche | coût CPU | coût humain | remarque |
|---|---|---|---|---|---|
| 1 | P2 (+ P1 comme calibration incluse) | NH-20 dans sa forme W₂/FR, sous-flots exacts | < 1 min | ≈ 6 h | réponse prévisible ; sert de calibration et de première ligne d'une note numérique |
| 2 | P3 | conditions d'identifiabilité C2-C4, Strang-JKO hors gaussien | 1 à 6 h | ≈ 8 h | seul dessin qui teste l'inférence C2 |
| 3 | P4 | ordre sur densité non gaussienne avec référence exacte | ≲ 2 h | ≈ 6 h | à faire si P3 laisse un doute sur la référence |
| 4 | P6 | prédiction chiffrée du défaut de Strang-JKO | < 1 h | ≈ 6 h | dépend de la lecture intégrale de Halmos-Hanin |
| 5 | P5 | explication rétrospective de M12 | 6 à 30 h | ≈ 8 h | campagne lourde ; seulement si une décision dépend de l'explication |

Le budget du trimestre pour NH-20 est de 10 h de verrou et une part des 15 h
de calcul (spec §5.2) : P2 + P3 tiennent dans cette enveloppe, pas P5.

## Ce qu'on ne fait pas, et pourquoi

- **Rejouer M12** avec plus de graines, d'autres γ ou une échelle de h plus
  fine sur la grille de 16 points : les conditions C2 à C5 resteraient violées,
  le résultat serait encore « non identifiable » par construction.
- **Revendiquer « Strang O(τ²) »** pour le JKO du labo : avec des sous-pas JKO,
  la prédiction est l'ordre 1 (note §3 ligne 4) ; l'ordre 2 demande des
  sous-flots exacts, d'ordre 2 ou appariés à leur adjoint.
- **Relier l'ordre aux opérations de consolidation** (REPLAY = transport,
  DOWNSCALE = réaction) avant P2/P3 : l'alternance elle-même est publiée
  (Gallouët-Monsaingeon 2016) ; l'interprétation ne vaut rien sans une mesure
  d'ordre propre.
- **2D ou GPU** : aucune des questions ouvertes n'en a besoin ; le 1D permet
  des références exactes.
- **Appliquer Carlier 3.16 comme garantie** pour le code du labo : ses
  hypothèses ne sont pas remplies (note §2bis), ni à ε = 0,01 ni à ε = h².

**Rappel : aucun de ces dessins ne part avant que le verrou soit passé.**
