# Context: moving people during the Dakar 2026 Youth Olympic Games

The Youth Olympic Games (Jeux Olympiques de la Jeunesse, JOJ) Dakar 2026 took
place from 31 October to 13 November 2026, the first Olympic event held in
Africa, with about 2,700 athletes aged 17 or under across three hubs: Dakar,
Diamniadio and Saly. Moving athletes, officials and spectators between these
hubs, without stopping the city, was one of the main operational challenges.

This page summarises the public information on the mobility plan, and explains
where this repository fits. Figures come from the sources listed at the end;
check the official documents before reusing them.

## Who organised mobility

- The Ministry of Land and Air Transport set up a **Mobility Committee for
  Dakar JOJ 2026** (Comité Mobilité Dakar JOJ 2026). Its secretariat was run by
  the **CETUD** (Conseil exécutif des transports urbains durables), Dakar's
  urban transport authority.
- **Dakar Dem Dikk** (DDD), the national bus company, was the official ground
  transport partner of the organising committee (COJOJ).
- A **Transport Operations Center (TROC)** was installed at CETUD on
  29 October 2026 to give a consolidated, real-time view of traffic during the
  Games. DDD had opened its own operational control centre in July 2026, with
  bus geolocation, on-board video and real-time incident reporting.

## The measures

| Measure | What was announced |
|---|---|
| Traffic plan | 166 traffic-calming measures and 14 junctions classed as critical or high risk; the plan applied around the sites from 17 October 2026, framed by prefectoral order no. 393 of 21 October 2026 |
| Public transport as backbone | TER (regional express train), BRT (bus rapid transit, in service since May 2024), Dakar Dem Dikk, AFTU minibuses, Games shuttles and accredited taxis |
| Games shuttles | 8 departure points and 12 links to the venues, 6:30 to 22:00, every 5 to 45 minutes depending on the line; DDD announced 184 buses for the shuttles |
| TER extension | the Diamniadio–AIBD airport section opened on 28 September 2026, linking the airport to Diamniadio and Dakar by rail |
| Olympic lane | on the Autoroute de l'Avenir (Patte-d'Oie–AIBD section), the left lane reserved for Olympic traffic towards Diamniadio from 7:00 to 19:00 |
| Heavy vehicles | restrictions on trucks around Diamniadio between 7:00 and 19:00 |
| Reserved and pedestrian streets | around Point E, rue du Tour de l'Œuf for pedestrians and accredited vehicles, two lanes of avenue Blaise Diagne for accredited buses |
| Corniche Ouest | temporary closures between 3 and 10 November, with varying hours |
| Information | the "CETUD Mobilités" mobile app grouping operators, routes and real-time alerts |

The authorities described the restrictions as "temporary, controlled and
reversible".

## Where computer vision helps

Most of these measures change **how many vehicles, of which type, pass a given
point**: a reserved lane should carry more buses and fewer cars, a closure
should move traffic to other streets, shuttles should replace private trips.
Manual counts are expensive and short; cameras already exist (bus fleets,
junctions, the TER). A counting tool turns that video into numbers that can
answer questions such as:

1. **Did the reserved bus lanes on avenue Blaise Diagne carry buses?**
   Count buses and cars per lane, per 5 minutes, during the Games.
2. **Where did Corniche Ouest traffic go during the closures?** Count flows on
   parallel streets on a closure day and on a comparable normal day.
3. **Did heavy vehicle restrictions hold?** Count trucks between 7:00 and
   19:00 near Diamniadio.
4. **What is the share of two-wheelers and minibuses** in the flow, a key fact
   for Dakar that generic traffic models often miss.
5. **What legacy?** Measuring the same points again after the Games shows
   which changes lasted, which matters for the next mobility policies
   (bus network restructuring, BRT and TER ridership).

`traffic-vision compare` is built for these before/during/after questions: it
compares the mean flow per interval between two recordings and gives a
bootstrap confidence interval for the difference.

## Limits and privacy

- **Comparable recordings**: compare the same place, same weekday and same
  time of day. A Tuesday morning and a Sunday evening differ for reasons that
  have nothing to do with a measure.
- **Detection errors**: occlusions in dense traffic, night video and vehicle
  types absent from training data (cars rapides, Ndiaga Ndiaye, carts) lower
  accuracy. Check a sample of counts by hand before drawing conclusions, and
  fine-tune on local images (`scripts/train_yolo.py`).
- **Privacy**: the tool counts classes and anonymous track numbers. It does
  not identify faces or number plates and keeps no image unless you ask for the
  annotated video. Filming public space in Senegal falls under Law no. 2008-12
  on personal data and the Commission de protection des données personnelles
  (CDP), which said the State's video-surveillance set up for the Games had the
  required authorisations. Any reuse of real footage should follow the same
  rules: aggregate data, limited retention, clear purpose.

## Sources

- Ministry of Land and Air Transport and CETUD, *Guide de la mobilité Dakar JOJ 2026* (1 October 2026), vie-publique.sn: https://www.vie-publique.sn/documents/14568/guide-mobilite-dakar-joj-2026
- Le Soleil, "JOJ Dakar 2026 : un plan de circulation déployé à partir du 17 octobre": https://lesoleil.sn/actualites/sports/joj-dakar-2026-un-plan-de-circulation-deploye-a-partir-du-17-octobre/
- Le Soleil, "une voie olympique et des restrictions sur l'autoroute de l'Avenir": https://lesoleil.sn/joj-dakar-2026/joj-dakar-2026-une-voie-olympique-et-des-restrictions-sur-lautoroute-de-lavenir-pour-la-ceremonie-douverture/
- Le Soleil, "un centre de contrôle installé pour une vision consolidée du trafic routier": https://lesoleil.sn/joj-dakar-2026/joj-dakar-2026-un-centre-de-controle-installe-pour-une-vision-consolidee-du-trafic-routier/
- Le Petit Journal Dakar, "la capitale se prépare à d'importants changements de circulation": https://lepetitjournal.com/dakar/joj-dakar-2026-la-capitale-se-prepare-dimportants-changements-de-circulation-456308
- Au Sénégal, "JOJ Dakar 2026 : comment se déplacer pendant les Jeux ?": https://www.au-senegal.com/joj-dakar-2026-comment-se-deplacer-pendant-les-jeux,18563.html
- Afrik.com, "vingt-cinq heures sans voitures sur la Corniche Ouest": https://www.afrik.com/joj-dakar-2026-vingt-cinq-heures-sans-voitures-sur-la-corniche-ouest-en-pleine-semaine
- Seneweb, CETUD plan for the Corniche Ouest: https://seneweb.com/en/news/Transport/joj-dakar-2026-le-cetud-devoile-le-plan-de-circulation-sur-la-corniche-ouest_n_999002564.html
- Le360 Afrique (AFP), TER extension to the airport, 28 September 2026: https://afrique.le360.ma/societe/senegal-mise-en-service-du-prolongement-du-train-express-de-dakar-jusqua-laeroport_7QGTAEREXBGD3PKZYHZL5GM4XI/
- AllAfrica / APS, Dakar Dem Dikk operational control centre (July 2026): https://fr.allafrica.com/stories/202607080245.html
- Seneweb, COJOJ and Dakar Dem Dikk partnership: https://www.seneweb.com/en/news/Video/dakar-2026-yog-coyog-gives-full-prerogative-to-ddd-to-be-the-official-ground-transport-partner_n_470130.html
- Xinhua via The Star, "Dakar 2026 Youth Olympics enters delivery phase": https://www.thestar.com.my/news/world/2026/02/06/dakar-2026-youth-olympics-enters-delivery-phase
- Osiris, "JOJ : le dispositif de vidéosurveillance mis en place par l'État est conforme à la loi": https://www.osiris.sn/joj-le-dispositif-de-videosurveillance-mis-en-place-par-l-etat-est-conforme-a.html
- International Olympic Committee, Dakar 2026 transport information: https://www.olympics.com/ioc/dakar-2026/transport
