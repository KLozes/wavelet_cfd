Commun.Comput.Phys. Vol.26,No.3,pp.654-680
doi:10.4208/cicp.OA-2018-0130 September2019
Fast Distance Fields for Fluid Dynamics Mesh
Generation on Graphics Hardware
AloRoosing1,∗, OliverStrickson1 and NikosNikiforakis1
1LaboratoryforScientificComputing,CavendishLaboratory,DepartmentofPhysics,
UniversityofCambridge,CambridgeCB30HE,UnitedKingdom.
Received4May2018;Accepted(inrevisedversion)10October2018
Abstract. We present a CUDA accelerated implementation of the Characteristic/S-
canConversionalgorithmtogeneratenarrowbandsigneddistancefieldsinlogically
Cartesiangrids.WeoutlineanapproachoftaskanddatamanagementonGPUsbased
on aninput of a closed triangulated surface with the aim of reducingpre-processing
and mesh-generation times. The workdemonstrates afastsigned distance field gen-
eration of triangulated surfaces with tens of thousands to several million features in
high resolution domains. We presentimprovementsto the robustness of the original
algorithmandanoverviewofhandlinggeometricdata.
AMSsubjectclassifications:68U05,68U20
Keywords: Signeddistancefield,GPU,CUDA,meshgeneration,fluiddynamics.
1 Introduction
Signed distance fields (SDF) find uses in domains from computer graphics [2] to nu-
mericalmodelling[3]. Determiningthelocationofexplicitorimplicitsurfacesingridsor
generatingmeshestodescribeobjectsisanareaofactiveresearchinmanycomputational
paradigms. Triangulated surfaces are a popular working medium and the Stereolithog-
raphy(STL)fileformatfindswideuseinareassuchasCFD[4]and3Dprinting[14]. The
quick generation of robust signed distance fields from triangulated surfaces is then of
greatinteresttomanyindustriesandacademicdisciplines.
Often it is necessary to know only the distance to the surface within a small region
aroundthegeometryandnarrowbandSDFsareusefulforquicklygeneratingjustthein-
tersectionbetweenacomputationalmeshandanobject. Thisfindsapplicationinembed-
ded boundary methods in computational fluid dynamics where generating object data
∗Correspondingauthor.Emailaddresses:ar694@cam.ac.uk(A.Roosing),ots22@cam.ac.uk(O.T.Strickson),
nn10005@cam.ac.uk(N.Nikiforakis)
http://www.global-sci.com/cicp 654 (cid:13)c2019Global-SciencePress

A.Roosing,O.T.StricksonandN.Nikiforakis/Commun.Comput.Phys.,26(2019),pp.654-680 655
oftentakesasignificantportionofthesimulationsetuptime,whichcanbecomeabottle-
neck in fast prototyping when the subsequent numerical work is highly optimised and
run on many-core architectures. For example, the signed distance field of a complex
carbody,asshowninFig.1,canbeusedtogeneratecutcellsinaregularcomputational
meshtoimposeboundaryconditionsalongadetailedperimeterwithoutintroducingsig-
nificantmeshgenerationoverheadorcomplexconnectivityinformation. Wefocusonthe
generationofnarrowbandsigneddistancefieldsinsideCartesiangridsbutthealgorithm
discussedinthispaperispotentiallyextendibletootherparadigms.
Ourmainaimistodescribearobustalgorithmtospeedupthegenerationoflevelsets
from triangulated surface information using graphics processing units (GPUs). In this
paperwediscusstheimplementationandadjustmentoftheCharacteristic/ScanConver-
sion (CSC) algorithm originally described by Mauch [5]. We will outline improvements
totheoriginalapproachandpresentanimplementationonGPUswithafocusonhowto
manageinformationaboutmanythousandsofconnectedfeatures.
Park et al. [7] have developed an algorithm for generating signed distances on the
GPU for hierarchical grids. They sample mesh cells based on the complexity of the sur-
face geometry and present a good speedup compared to identical approaches on the
CPU.Theiruseofangle-weightedpseudonormalsatsurfacediscontinuitiesissimilar to
thestrategyweemploy.
Sud et al. [11] describe a GPU signed distance field method based on Voronoi cells
and slicing. Their speedup stems from the use of GPUs, culling far away features and
clamping the rasterisation of the Voronoicells. Though their approach is different from
ours, the strategy of reducing calculations is similar to the current work. Their method
doesnotstoreinformationabouttheconnectivityoftrianglesandusestheCSCalgorithm
for suitable sub-problems, developing a new approach for problematic surface configu-
rations. OurimplementationispurelyCSCbasedandaddressesmanyofthesegeometric
cases.
Sigg et al. [10] presenta GPU implementation of the CSC algorithm for triangulated
surfaces. Their work is focused on overcoming the need for vertex extrusions by com-
bining edgeandfaceextrusions. Thisisdoneinordertoreducetheworkloadaswellas
avoidtopologicalcaseswhichtheCSCalgorithmfindsproblematic. Belowwediscussa
differentmethodologyfortheissuesarisingatvertices.
An implementation of the CSC algorithm also exists by Mauch [20]. We use some
of the insights of that code but have developed an independent strategy with updated
featuregeneration,ahighdegreeofparallelismandalgorithmicimprovements.
There is a lack of discussion in existing literature about how to best organise STL
features for use with the CSC algorithm on GPUs. Specifically, it is not immediately
clear how to efficiently produce extrusionsfrom nearby surface triangles when no strict
feature order is imposed in the input file. There are also gaps in the literature when
it comes to discussing some complex cases that can arise in common geometries such
as saddle vertices and other configurations discussed below. The main contributions of
thispaperaredescribingtheefficienthandlingofSTLfeaturesonGPUs,showingrobust

656 A.Roosing,O.T.StricksonandN.Nikiforakis/Commun.Comput.Phys.,26(2019),pp.654-680
Figure1: TheproducednarrowbandsigneddistancefieldandresultingsurfaceplotoftheDrivAercarmodel[15].
Complex geometries can be processed quickly to generate embedded meshes in the initial phases of fluid
simulations. Only creating a small shell around the underlying STL model is sufficient to describe a surface
intersecting a Cartesian mesh. The large number of surface features are ordered and used to build extrusions
which limit the space where distance calculations are made. Due to the short run times, high resolution
computationaldomainscanbeusedinconjunctionwithdetailedmodels,resultinginsophisticatedCFDmeshes
with regular memory layout.
extrusionbuildingforpreviouslyunaddressedsurfaceconfigurationsanddemonstrating
fastnarrowbandSDFgenerationforavarietyofcomplextestgeometries.
2 Closest point distance transform
The closest point distance transform algorithm [5] aims to populate domain cells in the
immediate vicinity of a geometry with the shortest distance to its surface. This is done
by generating individual fields from triangulated surface features and combining them
into a global signed distance field. Fig. 2 shows the input and output of the algorithm.
The initial data is a collection of triangles in 3D which describe a discontinuous surface
(Fig.2(a)). InatargetCartesiangrid,theCSCalgorithmpopulatesthecellsinthevicinity
ofthesurfacewiththesmallestdistancetotheobjectleadingtoanimplicitdescriptionof
thegeometry(Fig.2(b)).
The CSC algorithm can be used to generate the exact signed distance function of a
surface within a regular grid. This function is defined at every point in the vicinity of
the surface and grows in magnitude in the direction of the normals of the surface. For
orientablesurfaces,thepositiveandnegativevaluesdivideadomainintotheinteriorand
exteriorofthesurface,withthesurfaceitselflyingat0. Letxbeapointinthedomain Rn
andlet∂Ωbethesurface. Asigneddistancefunction f isthendefinedas:
f(x)=min{d{x,∂Ω}}, ∀x∈Rn, (2.1)

A.Roosing,O.T.StricksonandN.Nikiforakis/Commun.Comput.Phys.,26(2019),pp.654-680 657
(a) DetailofDrivAerSTLfile (b) ProducedsurfaceplotandSDFslices
Figure 2: Narrow bandSDF results for theDrivAer geometry. Thequickly generated field extends toa limited
distance from the surface. As the SDF is free from gaps, the 0 crossing matches the STL input to within a
fraction of ∆x.
whered{}givesthedistancebetweenapointandthesurface.
Forsmoothsurfaces, f(x)satisfiestheEikonalequation
|∇f|=1. (2.2)
Inthecaseofdiscretisedsurfaces,however,therearediscontinuitiesattheboundariesof
thesurfacefeatures. Inthiscase,thesigneddistancefieldofthesurfaceisthesumofthe
signeddistancefieldsofallsmoothregionsofthesurface.
TheCSCalgorithmusesthefeaturesofdiscretesurfacestogenerateextrusionsintheir
normal direction that are guaranteedto include at least the closestpoints totheoriginal
features. Theseextrusions are similar to Voronoi cells with the difference that they may
include more than the closest points to a feature, they are artificially enlarged and may
overlap. Thesumoftheseextrusionswillincludealltheclosestpointstothesurface.
Letd betheminimumdistancefromthemeshcellc tothesurface. Byconstructing
ijk ijk
extrusionsEforallofthefeaturesofthesurface,theCSCalgorithmcanbewrittenas:
{d =∞foralli,j,k }
ijk
foralle∈Edo
forallc ∈edo
ijk
d =distancetofeature
new
if|d |<|d |then
new ijk
d =d
ijk new
endif
endfor
endfor

658 A.Roosing,O.T.StricksonandN.Nikiforakis/Commun.Comput.Phys.,26(2019),pp.654-680
(a) STLfeature (b) face (c) edges (d) vertices
Figure 3: STL surface features. The file elements are divided into three aspects and each feature is used to
generate extrusions where theclosest distance to thesurface lies on thefeature.
Calculating the minimum distances to the surface from the mesh cells within all of
theseareasproducesasigneddistancefield. Thisoperationiscalledadistancetransform
and results in an implicit description of the surface within the rectilinear mesh. As the
workdone is boundedby thenumber ofsurface features and the number of cells in the
extrusions, the computational complexity of the algorithm is optimal: linear in both the
featurecountandtheresolutionofthemesh.
The CSC algorithm is limited to orientable closed surfaces. Thesegeometrieshave a
well-definedinteriorallowingforasigneddistancefieldwherethepositiveandnegative
distancesare oneithersideofthesurface, whichlies atthe0levelset. Thecurrentwork
is concerned with closed triangulated surfaces in 3D. The features of these surfaces are
thetriangularfaces,thetriangleverticesandthetriangleedgesasshowninFig.3.
3 Extrusions
Theextrusionpolyhedrafromthesurface featuresencompasstheareawheretheSDFis
calculated. We list the different extrusion types, how they are generated and how our
implementationdivergesfromtheoriginaldescription. Wedescribethecategorisationof
surface features and how this is used to reduce the amount of calculation that needs to
bedone,discussingunaddressedscenariosandproposedimprovements.
The CSC algorithm describes the construction of extrusions containing at least the
closest points to the discrete features. These extrusions are constructed based on the
position, limits and normals of the underlying geometries. Extruding outward from a
face produces a prism in the normal direction (Fig. 4(a)). An edge extrusion is a prism
extruded from the line between two vertices in the directions of the two neighbouring
faces(Fig.4(b))andavertexextrusionisapyramiddefinedbythenormalsoftheadjacent
facesthatmeetatthevertex(Fig.4(c)).
While the two prism extrusions have a known number of faces, the vertex pyramid
can be of arbitrary complexity which makes implementation and workload assessment
difficult. We simplify the vertex extrusion by using a cone that encompasses all of the

A.Roosing,O.T.StricksonandN.Nikiforakis/Commun.Comput.Phys.,26(2019),pp.654-680 659
(a) faceextrusion (b) edgeextrusion (c) vertexextrusion (d) vertexcone
Figure4: Theextrusionsfromdifferentsurfacefeaturesaregeneratedinthefacenormaldirections. Thevertex
extrusions can be simplified by assigning a cone which encompasses all of the face normals meeting at the
vertex.
Figure 5: The average pseudonormal, which is used as the axis of the cone, is constructed by weighting face
normalsbytheirrespectiveangles α. Theweightingdealswithissues arisingfrom manycoplanarfacesskewing
theaverage normal.
normals ofthefaces that meetat thevertex (Fig.4(d)). The newvertex extrusionis con-
structedin theaverage direction ofthe normals weightedby the angle betweenthetwo
edgesofeachtrianglethatmeetatthevertexinquestion. Takingtheunweightedaverage
canleadtoincorrectextrusions. AsdescribedbyBærentzenandAanæs[1],manycopla-
narfacessharingavertexcanshifttheaveragedisproportionallyawayfromwhatwould
be the intuitive direction of the vertex. The result is an angle-weighted pseudonormal
that correctly points in the average direction of the vertex (Fig. 5). This direction will
be the axis ofthe cone and theadjacent normal mostdiverging from theaverage lies on
the side of the cone. This way the obtained extrusion will include all the points inside
theoriginal cone and has a simple definitionofpointsinside and outsideit. This fix fits
well with the philosophy of the original algorithm where extrusionscontain at least the
closestpointsandonlytheminimumvalueisrecorded.
3.1 Surfacecurvature
Thesigneddistancefunctiondescribesbothpositiveandnegativevalues. Theextrusions
must then be constructed on both sides of the surface features. We adopt the conven-
tion that the interior of the surface is negative and the exterioris positive. The outward

660 A.Roosing,O.T.StricksonandN.Nikiforakis/Commun.Comput.Phys.,26(2019),pp.654-680
(a) convexedge (b) concaveedge (c) convexvertex (d) concavevertex
Figure6: ThesurfacecurvatureisdefinedbytheSTLsurfacefeaturesandaveragenormals. Convexedgesand
vertices have all of their neighbouring vertices below their average normal plane. Concave features have all of
theirneighbours above theplane.
extrusions are then along the feature normal and the inward extrusions in the negative
normal direction. For all the triangle faces, prism extrusions extending in both positive
andnegativedirectionswillencompasstheareaclosesttothatface. Workcanbereduced,
however,fortheedgeandvertexcasesbasedonthecurvatureofthelocalsurface. Mauch
introducestheconceptsofconvexandconcavefeatures.
Takingtheplane definedby thepositiveaveragenormal offaces meetingat an edge
andoneoftheedgevertices,anedgeisconvexifitsendpointslieaboveitsneighbouring
points,concaveiftheyliebelowandflatotherwise(Figs.6(a)and6(b)). Thesameistrue
for the coordinates of a vertex, which is convex when its neighbouring vertices all lie
belowtheplanedescribedbythevertexandtheangleweightedaveragepseudonormal.
A vertex is concave when its neighbours all lie above that plane, and flat when they all
lie on the same plane (Figs.6(c) and 6(d)). Convex features will then only needpositive
extrusions,concaveoneswillonlyneednegativeonesandflatregionswillneednoneas
thesurroundingextrusionsfromotherfeaturesfillthearea.
3.2 Saddle
Aspecial case exists,however,whereavertexis neitherconvex, concave nor flat. These
saddlepointsoccurincommongeometriesandtheoriginalalgorithmdoesnotdealwith
the gaps left in-between the other extrusions, leading to regions of undefined distances
andan incorrect signeddistancefield. Asaddlepointoccurswhenthereareneighbour-
ing vertices both above and below the plane described by the average pseudonormalof
a vertex and the vertex itself as shown in Fig. 7(a). A fix for this problem, as suggested
by Peikert and Sigg [8], is to use both a positive and a negative extrusion at these ver-
tices. Thesespecialcaseswillthenwarrantdoubletheworkloadofothercurvaturesbut
weobservethatthisstrategyfully coversthevolume aroundthevertexin all ofourtest
casesandleadstoaconsistentsigneddistancefieldaroundcomplexdiscontinuities.
A question then arises about the shape of the extrusion from a saddle point vertex.
While smooth convex/concave regions create a convex gap that is limited by the nor-

A.Roosing,O.T.StricksonandN.Nikiforakis/Commun.Comput.Phys.,26(2019),pp.654-680 661
(a) saddlevertex (b) ruffgeometry
Figure 7: High curvature geometry. Saddle points are vertices where there are neighbouring vertices on both
sides of the pseudonormal plane. Ruff geometries can be convex, concave or saddle vertices which feature
normals that point to more than the half-space around the pseudonormal plane. These lead to complex gaps
between extrusions from other features or call for extrusions that cause sign ambiguity.
mals ofadjacent faces, in saddlepointcasesthis volume can becomplex and difficult to
assignanextrusionto. Byusingaconedefinedbythepseudonormalastheaxisandthe
mostdivergingnormalonitsside,therelative orderand configurationofothernormals
doesnotmatter andwe have awellformedvertexextrusion. Forsaddleshapesourap-
proachistofirstgeneratetheconeforthepositivesideandthenreflectitinthenegative
pseudonormaldirectionfortheinteriordistancegeneration.
3.3 Ruff
Even fully convex/concave vertices can have normals which do not define a simple re-
giontoextrudeinto. ConsiderthecaseshowninFig.7(b). Theillustratedruff-likeshape
isavalidorientabletriangulatedsurfacewherefaceswithalmostoppositepointingnor-
mals meet at a single vertex. As all the neighbouring points end up being on the same
side of the pseudonormal plane, the vertex is classified as convex. However, the space
enclosed by the sum of the face normals extends below the pseudonormal plane at the
vertex. Similarly, saddle points often, but not always, feature a collection of normals
spanningmorethanthehalf-space.
Asimplesolutionforthesecasesistoonlyconsidernormalspointingtothesameside
of the pseudonormalplane as the average normal itself. The volume bounded by these
positive pointing normals will be strictly less than the half-space above the pseudonor-
malplanewhichiscoverablewithacone. Forourtestcasesthisstrategyfillstheregions
of ruff shapes and producessigned distance fields consistent with the input surfaces. It
ispossible,however,thataconeencompassingonlypositivepointingnormalsisnotsuf-
ficient to cover thespace betweenface and edgeextrusions. In such cases a hemisphere
canbeconstructedinthepseudonormaldirectiontocovertheentirehalf-spaceabovethe
vertex.

A.Roosing,O.T.StricksonandN.Nikiforakis/Commun.Comput.Phys.,26(2019),pp.654-680
662
| 4   | Completeness | of  | the CSC | algorithm |     |     |     |     |
| --- | ------------ | --- | ------- | --------- | --- | --- | --- | --- |
Weshowwhytheabove mentionedprocedurescover theimmediate vicinity ofthesur-
facewithoutleavinganygapsandwhysignconflictscanberesolvedunambiguously.
R3,sothattherearenoholes.
ConsidergeneratingtheSDFforallof Spaceisdivided
bythesurfaceintotworegions,insideandoutside,wheremovingfromoneregiontothe
other along a continuous path necessitates crossing the surface. For any point outside,
theclosestpointonthesurfacetothispointcouldlieoneitheraface,anedgeoravertex.
Considerthesimplifiedcaseofasinglevertex,withedgevectorsextendingtoinfinity.
Thevertexisattheorigin,andthenormalizededgevectorsarelabelledv ,···,v (Fig.8).
1 n
Wedisallow faces ofzeroarea(adjacent edgevectorsthatare parallel orantiparallel). A
faceextrusionfromtheith
face,isthenthesetofpointsgivenby
|              |                                       |     |         | +µv   | +ν(v     | ×v ), |     |       |
| ------------ | ------------------------------------- | --- | ------- | ----- | -------- | ----- | --- | ----- |
|              |                                       |     | λv      | i i+1 |          | i i+1 |     | (4.1) |
| withλ,µ,ν≥0. | Theneighbouringedgeextrusionisgivenby |     |         |       |          |       |     |       |
|              |                                       |     | λv +µ(v |       | ×v )+ν(v | ×v    | ),  | (4.2) |
|              |                                       |     | i       | i−1   | i        | i i+1 |     |       |
withλ,µ,ν≥0.
|     |     |     | Figure | 8: Edgevectors |     | at a vertex. |     |     |
| --- | --- | --- | ------ | -------------- | --- | ------------ | --- | --- |
The vertex extrusion is the positive spanning set of the surrounding face normals,
namely,thesetofpointsgivenby
∑
|     |     |     |     |     | λn, |     |     | (4.3) |
| --- | --- | --- | --- | --- | --- | --- | --- | ----- |
|     |     |     |     |     | i i |     |     |       |
i
| wheren | =v ×v   | /|v ×v | |,n   | =v ×v | /|v | ×v |,andλ | ≥0foreachi. |     |
| ------ | ------- | ------ | ----- | ----- | --- | --------- | ----------- | --- |
|        | i i i+1 | i      | i+1 N | N     | 1 N | 1         | i           |     |
R3
| Apositivespanningsetofvectorsin |     |     |     |     | iseither: |     |     |     |
| ------------------------------- | --- | --- | --- | --- | --------- | --- | --- | --- |
1. aninfinite,convexpyramid,whoseedgesaretheconvexhullofthevectors,
2. aninfinitewedge,whentwoofthevectorsareantiparallel,
| 3.  | ahalf-spaceof | R3, |     |     |     |     |     |     |
| --- | ------------- | --- | --- | --- | --- | --- | --- | --- |
| 4.  | theentiretyof | R3. |     |     |     |     |     |     |
Asetoffacenormalscan resultinanyoneofthesecases. Thelattertwocasescanbe
obtainedfromruffgeometries.

A.Roosing,O.T.StricksonandN.Nikiforakis/Commun.Comput.Phys.,26(2019),pp.654-680 663
For the SDF generation to be correct, these extrusionsmust fill the space to one side
of the surface completely, since each extrusion is a superset of points where that edge,
faceorvertexistheclosestpointonthesurface,andeverypointinspacehasatleastone
closestpointtothesurface,whichmustlieonsomefeature.
From the above, it is clear that the procedure will not lead to any gaps: the edge
extrusions are defined by the sides of the face extrusions, without any space between
them, and thevertex extrusionsare definedby thespan ofnormals of thefaces meeting
at a vertex, the convex hull of which will always include all of the normals. There are
nootherfeaturesofatriangulatedsurfaceandintheabsenceofgapsinaclosedsurface,
everypointinitsvicinitymustexistinanextrusion.
4.1 Conflicts betweenpositive and negative extrusions
Itispossibleusingtheproceduredescribedaboveforagivenpointofthedomain(either
inside or outside) to be both in a positive and a negative extrusion. Only one of these
can be of thecorrect sign, dueto the orientability of thesurface. Choosing the extruded
distancefieldwithminimumabsolutevalueatthepointinquestionisenoughtoresolve
themajorityoftheseconflicts. Wedescribeeachcasebelow.
4.1.1 Incorrectdistanceinformationfromfacedata
Consider a point strictly on a face (not an edge or a vertex). As the surface normals n
N
point outside, there is some ǫ where ǫn is within the exterior of the surface (i.e. there
i
is always free space immediately adjacent to a face in the normal direction). Thus, if a
sign conflict arises from a face extrusion at a point p, then a face extrusion must have
crossedthesurface. Consider Fig. 9(a): let thevalue of theincorrect extrusionfrom face
f have absolute value D, and let the shortest distance from our point p to c where the
extrusioncrossedthesurfacebed,thenD > d+ǫ. Thismeansthatthereisacloserpoint
to p on the surface whose extrusion has the correct sign and the conflict does not cause
anyambiguity.
4.1.2 Incorrectdistanceinformationfromedgedata
Thesituationis notassimpleforedgevectors,sincetheface normalsboundingan edge
extrusion may point into the surface. However, note that if the edge extrusion with in-
correctsignisgivenby
{λv +µ(v ×v )+ν(v ×v ):λ≥0,µ,ν > 0} (4.4)
i i−1 i i i+1
thentheedgeextrusionofthecorrectsignis
{λv −µ(v ×v )−ν(v ×v ):λ≥0,µ,ν > 0}. (4.5)
i i−1 i i i+1
Thesesetsaredisjoint(Fig.9(b)),meaningthatiftheincorrectedgeextrusionconflicts
withanotherextrusionoftheoppositesign,theclosestpointonthesurfacecannotbeon

664 A.Roosing,O.T.StricksonandN.Nikiforakis/Commun.Comput.Phys.,26(2019),pp.654-680
(a) incorrectsignatface (b) incorrect sign at (c) ambiguous sign at (d) assigning a
whenanextrusionfrom point p from the edge vertex caused by nor- hemisphere in the
facefextendstopointp e will be overridden by mals spanning all of pseudonormal direc-
where the ambiguity is anotherextrusionwhere R3 which will lead to tions in such cases is
resolved by the smaller theedgeecannotbethe the positive and nega- enough to generate two
distancedfromanearby closestfeatureasitstwo tive extrusions to over- disjoint volumes where
face which the incorrect extrusions of different lap the closestpoints inside
extrusion must cross at signsaredisjoint extrusions are dealt
pointc withcorrectly
Figure 9: Sign ambiguity at features.
that edge at all, and there will exist an extrusion (from a face, vertex or another edge)
with smaller absolute value. In other words, in cases where an edge extrusion would
assignthewrongsigntoapoint,thatedgecannotbetheclosestfeaturetothatpoint,and
intheabsenceofgaps,thepointwillalsofallwithinsomeotherextrusionwithasmaller
magnitudeandthecorrectsign.
4.1.3 Incorrectdistanceinformationfromvertexdata
Aswiththeedgedata,itispossibleforthefacenormalstopointintothesurface: thatis,
forǫn tobeintheinteriorofthesurfaceforallǫ > 0.
i
Forthefirstthreecasesdescribedaboveforvertexextrusions,theyhavetheproperty
thatthecorrespondingextrusionofoppositesignisdisjointfromtheoriginalone. Similar
to the case of the edge extrusion, this means that in the case of conflicting information
due to the propagation of an incorrect sign from the vertex, there is a closer point from
anotherextrusionofthecorrectsign.
Forthefinalcasewherethefacenormalsspanthedomain,thereisagenuineambigu-
itybetweenthepositiveandnegativeextrusions: theyarebothpropagatinginformation
withthesameabsolutevalueofthedistance,butwithconflictingsigns(Fig.9(c)).
We solve the ambiguity by first computing an angle-weighted pseudonormal at the
vertex. BærentzenandAanæs[1]showthatthispseudonormalcanbeusedasadiscrim-
inantforthesurfaceatthevertex: ifpisapointwhoseclosestpointonthesurfaceisthe
vertex,thenN ·p> 0whenpisoutsideofthesurfaceandN ·p< 0whenpisinsidethe
α α
surface,where
N
=∑
α n (4.6)
α i i
i

A.Roosing,O.T.StricksonandN.Nikiforakis/Commun.Comput.Phys.,26(2019),pp.654-680 665
isthepseudonormal,andα istheangleofthefacewithnormaln.
i i
TheextrusionisthenperformedonlyforthehemisphereorientedintheN direction
α
forthepositiveextrusion,andinthe−N directionforthenegativeextrusion(Fig.9(d)).
α
The positive and negative extrusions are disjoint, apart from the plane normal to N .
α
Theclosestpoint lying exactly on the plane can be excludedas if p is on the plane, then
N ·p=0, and so by the discriminant property, p is on the surface and so is the vertex
α
itself.
ToshowthatthisdoesnotresultinanygapsintheSDF,noticethatapointoutsideof
the positive hemisphere has N ·p< 0, and so either p is closer to a point on the surface
α
otherthanthevertex(andsomustbelongtoanotherextrusion),orpis intheinteriorof
thesurface.
4.2 Cone extrusion of vertices
For cases where the normals at a vertex describe an infinite convex pyramid, we use a
supersetof the normals instead. The supersetis formed by first constructing the angle-
weighted pseudonormalN as described above, finding the face normal n at the vertex
α
which divergesmostfromit asi =argmin|n ·N |,whereargmin givestheargument
min i α
i
whichminimisestheresultandconstructingaconewiththisnormallyingonitsside.
Thepseudonormaliswithintheoriginalconvexpyramid,sinceitisapositivecombi-
nationofthenormalvectors. Sincen minimizedtheright-handsideofthisexpression
imin
amongthen,theothern arecontainedwithinit,asistheoriginalpositivespan,since
i i
N ·∑ cn =∑ c N ·n >∑ cN ·n =N ·n (4.7)
α i i i α i i α imin α imin
foranypositivecoefficientsc withunitsum.
i
The above shows that the produced distance field will not have any gaps and that
sign conflicts can be dealt with unambiguously. When limiting thealgorithm to narrow
bands,thesameholdstrueastheextrusiondistancesarethesameforallfeatures.
5 Scan conversion
Aftergeneratingtheextrusions,weneedtodeterminewhichdomaincellslieinsidethem.
Thisisasimilarproblemtoscanconversion–amethodincomputergraphicsthattrans-
forms mathematically described polygonsinto rasterisedshapes. Mauch describes how
we can determine which discretely spaced cells are inside continuous extrusions in 3D
byreducingthescanconversionofapolyhedrontoaseriesof2Dproblemswhereslices
oftheextrusionsarescanconvertedonplanesofmeshcells.
However,theextrusionsofthesurfacefeaturesarealwayseithercones,hemispheres
orconvexprismswhichcanberasterisedin3D.Fortheprisms,werasterise3Dregionsof
themeshbasedonthehalfplanetest. Thisstrategyisusedtofindoutifapointiswithin
a convex polyhedron by determining if it is on the same side of all of the polyhedron

666 A.Roosing,O.T.StricksonandN.Nikiforakis/Commun.Comput.Phys.,26(2019),pp.654-680
faces. Wefindthatthischangefitsbetterwiththeoverallstrategyofmultithreadedcom-
putationandgivesrisetootheroptimisationtechniquesdescribedintheimplementation
section.
Letc beacellinthedomain DandletEbeanextrusionwithinwardpointingface
xyz
normals N. Furthermore,let p be a point on theith face of an extrusion. Thehalf plane
i
testtodetermineifapointiswithinaconvexpolyhedroncanbewrittenas:
forallc ∈D do
xyz
foralln ∈N do
i
if(n ·c −p)< 0then
i xyz i
returnfalse
endif
endfor
returntrue
endfor
Intheimplementationthescalarproductistestedagainstsomesmallvalueǫ,totake
intoaccountnumericalerrorsandthespacingofCartesiangridcellcentres.
Apointiswithinahemisphereifitiswithinaspecifieddistanceofthespherecentre
andonthecorrectsideofaplane. Pointsinsideaconesatisfythecondition:
N ·(p−v)
α >N ·n , (5.1)
|p−v| α md
wherep is the point, v is the vertex, N is the unit cone axis and n is a unit vector on
α md
thesideofthecone.
6 Implementation
InthefollowingsectionsweoutlinethetechnologicalparadigmofGPUsandourimple-
mentationincluding datastructures,geometrygenerationandtheworkschedulingand
calculation of the signed distance field at discrete intervals. The code takes as input a
stereolithogrpahy(STL)file and outputsafile that lists the SDF values in a 3D grid. We
are interested in generating a signed distance only at the immediate vicinity of the ge-
ometryandarethenonlyconcernedwiththecellsinsidetheunionoffeatureextrusions
extendingtosomesmalluser-definedmaximumdistancefromthesurface.
The CUDA programming platform is a C-like interface by Nvidia for programming
GPUs[6]. TheSIMDarchitecturefitswellwithaCartesiangriddatastructurewithmini-
maldependencebetweendifferentpartsofanalgorithm. CUDAallowstheprogrammer
to launch a large number of threads that are scheduled and executed on the graphics
card. Thoughverypowerful,GPUsrequireastrategydifferenttoconventionalCPUcod-
ing. The main speedup of CUDA comes from having a large bank of threads and fast
contentswitchingtomaskresourcefetching.

A.Roosing,O.T.StricksonandN.Nikiforakis/Commun.Comput.Phys.,26(2019),pp.654-680 667
Figure 10: Schematic of some CUDA concepts. The SDF domain will reside in global memory and kernels are
launchedto generate thesigned distancevalues. A large numberof threadsgrouped intoblocks allow for high
parallelism. There are limited banksof memory shared in blocks and smaller registers for each thread.
Theprogrammercanlaunchblocksofthreadsthatarehandledandscheduledbythe
GPU and execute algorithmically simultaneously, but in practice, partially sequentially
in an unspecified order. The threads are grouped into warps of 32 which execute the
sameinstructionsimultaneously,andinthecaseoflogicalbranching,parallelismmaybe
lost. Thereisabankofslowaccessglobalmemoryavailable toallthreads,blockspecific
shared memory and thread specific registers and limited scope caches. Fig. 10 shows a
conceptualdiagramofthedifferentmemoryspacesandgroupingsfromaprogrammer’s
pointofview.
Memory access is very important to getting good performance, as the difference in
bus speedsbetweenthe main global memory compared to registersand caches is many
orders of magnitude. Conventional CUDA codes make use of programmer specified
cachingandread/writecoalescencewhereunitsofthreadsaccessmemoryclosetogether,
therebyreducingthenumberofpagetransferoperations. Thesparselayoutofasurface
in3D, however,givesrisetosomeinterestingquestionsaboutmemoryaccessandwork
scheduling. This stems from data no-longer lying adjacent in logical groups in the do-
main, but along an arbitrary surface. It is not therefore immediately clear how to best
addressmemoryinawaythatwouldminimisefetchtransactions.
Inthefollowingsections,wewilldescribeourapproachtotheoriginalalgorithmand
thecodethatwasproduced. Theimplementationencompasseseverythingfromreading
intheSTLfiletooutputtingaresultfile. However,weonlytimetheworkdonecreating

668 A.Roosing,O.T.StricksonandN.Nikiforakis/Commun.Comput.Phys.,26(2019),pp.654-680
the internal geometry representation and the SDF generation. We will assume that the
STL file describes a correct closed surface with no gaps between adjacent faces and no
overlappingorflippedfaces.
7 Data structures
TheimplementationstartswithreadinginthestructuredSTLfilethatliststheverticesof
eachtrianglefaceinacounterclockwisedirectionandanoutwardpointingnormal:
facet normal ni nj nk
outer loop
vertex v1x v1y v1z
vertex v2x v2y v2z
vertex v3x v3y v3z
endloop
endfacet
This information is used to construct a single entry in a Face object, three entries in
an Edge object and three entries of a Vertex object. These objects are collections of the
spatialcoordinatesoftheverticesandnormalsofthefeatures. Welistthemasstructures
ofarrayswhereallthexcoordinatesarefollowedbyalltheycoordinatesandfinallythe
zcoordinates. WegeneratetheseobjectsontheCPUandcopythemintotheGPUglobal
memory. While a Face object fully describes a triangle with a normal, Edge and Vertex
objectsneedfurtherprocessingtogenerateextrusions.
7.1 Edge data
AnEdgeobjectis createdwith twoend-pointsofanedgeandanormalofthetriangleit
was constructed from which is insufficient for an edge extrusion which needs two nor-
mals. Assuming a correct closed surface, there exists another entry with identical end
points but a different normal. We would like to find matching pairs of edge features in
the fastest possible way without checking each pair of endpoints against all the others.
AstheorderoftrianglesinanSTLfilecanbearbitrary,wewouldliketoordertheentries
intheEdgeobjectsuchthatthepairsarenexttoeachother.
Sorting points in 3D has no one correct solution, more so for pairs of points. One
approach is to generateMorton codes for all of the points. Working with 32 bit floating
pointvaluesforallofthecoordinatevalues,wecan generate30bitintegervaluescalled
Morton codes for each 3D point. These values will retain their relative position when
sorted. Specifically, the sorted Morton codes will produce a Z-curve ordering. For our
purposes, the actual order does not matter, only that identical edge features are posi-
tionedconsecutively.
AnintegerMortoncodegeneratedfromthethreefloatingpointcoordinatevaluesof
avertexwilldesignateitspositionina1Darray. Weusea30bitintegervaluestoredina

A.Roosing,O.T.StricksonandN.Nikiforakis/Commun.Comput.Phys.,26(2019),pp.654-680 669
Figure 11: Expanding a 10 bit variable in four steps. The figure illustrates the movement of the original bit
position values in thevariable. (Adapted from [9])
Figure 12: Interleaving three bit-patterns into a single 32 bit int variable using OR operations. (Adapted
from [9])
32 bit intvariable with thetwohighestbits setto0. The32 bit floatcoordinatevalues
are first bit shifted to give 10 digits preceding the decimal point. We then expand the
threevaluesusingbitwiseoperationsasshowninFig.11ormoreconciselyincode:
x = (x | (x << 16)) & 0x030000FF;
x = (x | (x << 8)) & 0x0300F00F;
x = (x | (x << 4)) & 0x030C30C3;
x = (x | (x << 2)) & 0x09249249;
TheresultingthreeintvaluesareusedtobuildtheMortoncodebyshiftingtheyand
zvaluesfurtherandinterleavingallthreeintoasinglevariableasshowninFig.12.
Theexpansionandinterleavingofthree10bitvalueslimitsusto10243uniquevalues.
Wespecify theMortondomain to encompassa spacethat is definedby thesmallest x, y
and z Vertex values at one corner and the largest value Vertex at the opposite extreme.
Welaunch athreadperEdgeobjectand storetheMortoncodesin an intpointeronthe
GPU.
AnEdgeentrywithtwoendpointscanthenbetransformedintoaunique60bitlong
value where two 30 bit int values are concatenated such that the larger value takes up
thehighbitpositions. Twoedgeswiththesameendpointsthenhavethesamecodeand
we can sort them to position identical edges next to each other. Because of the limited
resolution of three 10 bit values, this may still lead to a case where several Edge pairs
havethesameMortoncode.
We use the Thrust [19] library to sort a list of integer position indices based on the

670 A.Roosing,O.T.StricksonandN.Nikiforakis/Commun.Comput.Phys.,26(2019),pp.654-680
Morton code map using thrust::sort by key. We then reorder the Edgeentries based
on the indices using thrust::gather. This produces groups of edge entries with iden-
tical Morton codes being consecutive. Finally, we launch a thread for each group and
sequentially traverse the collection with identical codes, reordering them if necessary
suchthatidenticaledgesareconsecutive. Inpractice,thisstillallowsforhighparallelism
butintheory,therecanbeanoticeabledifferenceintheamountofworkeachthreaddoes
whentheunderlyinggeometryhaslargevariationinthesizeoftriangles.
7.2 Vertexdata
The entries of the Vertex object are similarly incomplete. Each entry has data about the
position of the vertex and the normal of the face it was generated from. In addition we
know the angle between the two edges connecting at the vertex on that face. We also
retain data about the two other vertices on the original triangle. We again employ the
MortoncodestrategyfromtheEdgeobject. Wegenerate30 bit intcodesfor each entry,
sortalistofindices,reordertheVertexobjectsandgroupidenticalentriestogether. This
leadstoanorderedlistofentrieswhereidenticalverticesareconsecutiveandeachretains
auniquenormalandangle.
8 Extrusion generation
Oncethedatastructureshavebeenprocessed,wecangeneratetheextrusions. Thereare
threetypesofextrusions: prismsfortheFaceandEdgeobjectsandconesorhemispheres
for the Vertex objects. A prism is defined by six points and five sides. However, in
order to tell if a point is within the area we are interested in, only four side normals
and two points are needed (either two vertices on a face or the end points of a edge).
A cone requires a point, an axis vector and the most diverging normal on its side. The
hemisphererequiresapointandaclippingplane.
8.1 Face extrusion
The prism from a face is constructed by first extruding the three corner vertices by the
userspecifieddistanceinthenormaldirection. Wethenfindsidenormalsdefinedbythe
cross product of the counter-clockwise ordering of the vertices when viewed from the
insideoftheprism. Thesethreenormalsandpointsonthesidescanbeusedtodescribe
the planes of the prism sides. We use two of the original face vertices as the points on
the planes. We then save the smallest and largest coordinate values of the original and
extruded vertices. This produces a cuboid axis aligned bounding volume (AABV) that
contains thecell centreswewish to testfor inclusion in theprism. Thesame is donefor
the negative extrusion of the original vertices in the flipped face normal direction. We
endupwithtwoPrismobjectsandtheirAABVs.

A.Roosing,O.T.StricksonandN.Nikiforakis/Commun.Comput.Phys.,26(2019),pp.654-680 671
8.2 Edge extrusion
Anedgeextrusionisalsoaprismbutextrudedfromaline. Westartbydeterminingifan
edgeisconvex,concaveorflat. Lettheedgepairbetweenverticesaandbbedenotedby
abandbawithnormalsn andn respectively. Wedefineadiscriminantd=(ab×n )·n .
1 2 1 2
Anedgeisconvexifd > 0,concaveifd < 0andflatifd=0.
Forconvexedgesweextrudethetwoendpointsoftheedgebythespecifieddistance
inboththenormaldirections,therebyproducingaprism. Thefoursidesoftheprismare
describedbytheendpointsandtheinwardpointingnormalsconstructedsimilarlytothe
face prisms. An AABV is also constructed to encompass the prism. For concave edges,
theoriginalnormals arefirstflippedandtherestoftheprocedureis identical. Forsome
geometries,itwasdiscoveredthataclippingplaneatthebaseoftheedgeextrusionwas
neededtoproduceasmoothsurfaceoutput. Theplaneisdefinedbytheaveragenormal
oftheedgeandoneoftheendpoints.
8.3 Vertexextrusion
Theregularvertexextrusionisaconewithacirclebase. Toconstructit,wefirstscalethe
normals ofthe vertex entriesby their angle. The collection of normals correspondingto
a single vertex are then used to generate an average pseudonormaland find the largest
anglebetweentheaverageandtheoriginalnormals.
All neighbouring vertices v are tested to see if they are above or below the plane
N
defined by the original vertex v and the pseudonormal N . Consider the discriminant
α
d=vv ·N . Vertexvisconvexifd > 0, ∀v ,concaveifd < 0, ∀v ,flatifd=0, ∀v anda
n α N N N
saddlepointotherwise.
Forconvex,concaveandsaddleshapeswestorethevertexcoordinates,thepseudonor-
mal and the most diverging positive pointing normal. Similarly to prisms we define a
bounding volume. The negative extrusion is constructed in the same way, but with a
flippedaveragenormal, wheresaddlepointsuseareflectedpositiveextrusion. Forruff-
likescenarios,wedefineanAABVofahemisphereclippedatthepseudonormalplane.
When constructing the cone, the height is the user defined maximum distance. For
sharpcorners,itmayhappenthattheconebaseisverylargeandifpositioneddiagonally
in the domain, would require a large AABVwhich would extendfar beyond the region
closesttothe vertex. Toavoid testingunnecessarilymany cells, we takethe intersection
of the AABV of the cone and the bounding volume of a sphere with the radius of the
maximum distance centred at the vertex. This leads to a smaller AABV and fewer cells
tocalculatetheSDFfor.
9 Work scheduling
Afteralloftheextrusionshavebeengenerated,wecometotheproblemofhowtosched-
ule the SDF generation. For best performance, we would like to limit the number of

672 A.Roosing,O.T.StricksonandN.Nikiforakis/Commun.Comput.Phys.,26(2019),pp.654-680
calculations and memorytransactionsand doas much workas possiblein parallel. The
mainvariablesinoursoftwarearethedomainresolution,thedesiredmaximumSDFdis-
tance and the number ofsurface features. Regardlessof theextentofthe computational
domain,weonlywanttocalculatetheSDFforthesumofcellsinsidealloftheextrusions,
whichoftenoverlap. Todetermineintersectionofasurfacewiththecomputationalmesh,
anSDFdistanceofaround5∆xissufficientwhere∆xisthelengthofacellinonedimen-
sion. To limit which cells check for inclusion in which extrusions, the code works only
onthecells insidetheboundingvolumes. Workis thereforeonlydoneonthecells most
likely to be within any extrusionand we limit the testedcell and extrusion pairs. There
aretwoobviousapproachestoparallelisminthiscase.
The first is to check each cell in a bounding volume simultaneously. The start and
end x, y and z coordinates of the volume and the resolution of the domain are stored in
theextrusiondata. Thenumberofcellsthevolumecoversisthenknownandthreadsare
launchedaccordingtothesizeofthevolumeandthedomaincoordinatesofeachthread
canbedeterminedfromthelimits oftheboundingvolume. Allthreadscheckiftheyare
within thebounding volume’s extrusionin parallel. For threads that are inside, the dis-
tance to the feature can be calculated, and threads with a smaller magnitude value than
the previous one write their result to memory. This leads to warp divergence but as no
actionistakenfortheothercases,thereisnoperformancepenalty. Thisimplementation
would launch a kernel per bounding volume where each thread works with the same
datawiththeexceptionoftheirlocalcoordinatedataandthedistancetheycalculate. For
narrow band SDF generation of objects with uniform feature sizes, the bounding vol-
umesare likely tobe small and for high featurecounts, thekernellaunch will dominate
theruntime,leadingtopoorscaling.
Thesecondapproachis toparallelise overthesurfacefeatures. A threadislaunched
per extrusion and it loops through each cell location within the bounding volume, de-
termining whether to write a distance value to memory. For narrow bands and high
featurecounts,theserialtraversalofboundingvolumecellsisrelativelylightweightand
fast. However,manyoftheextrusionsoverlapandtheimplementationmustensurethat
the smallest magnitude value is found. For parallel computation, the writing must be
atomic,whichwillintroducesomeserialisationwhenmultiplethreadsareworkingwith
thesamedomaincoordinates. WeusetheatomicCASmethodtotrytowriteafloatvalue
into memory if the recorded value at the address has a larger magnitude. This attempt
continuesuntilthelocalvalueissuccessfullywrittentomemoryorasmallermagnitude
value is written by another thread. The effect of the serialisation depends on the input
geometryandthethreadschedulingbuttheimpactontheruntimeissmallcomparedto
theoverallamountofwork.
9.1 Dynamicparallelism
Consider,however,geometrieswithfew featuresin highresolutiondomains (e.g. when
simulating flow over a box). When the number of cells inside extrusionsis significantly

A.Roosing,O.T.StricksonandN.Nikiforakis/Commun.Comput.Phys.,26(2019),pp.654-680 673
higherthanthefeaturecount,loopingovercellsinsideboundingvolumesdominatesthe
runtime. While the overall generation time is usually on the order of seconds, there is
still scope for improved performance by using a hybrid of the two approaches outlined
above. Dynamic parallelism allows for kernels to be launched from the device. Wang
and Yalamanchili [13] provide an analysis of CUDA dynamic parallelism. They show
that there is potential for speedup in several problems with inhomogeneous workload
butthatthegreateroverheadoflaunching kernelsonthedevicecannegatethebenefits.
Tang et al. [12] discuss a dynamic platform which seeks to launch device side kernels
only when the potential computation time outweighs the launch overhead. They show
good speedupfor several benchmark problems. A hybrid approach would then launch
asinglekernelfromthehost,assigningasinglethreadforeachboundingvolumewhich
dynamicallylaunchkernelswithathreadpercell.
Launching kernels on the device has a greater overhead than host side launches but
dynamicparallelismallowsformoreworktobedonesimultaneously. Wethereforecon-
sider two alternatives: launching a thread per extrusion to loop through the cells or
launching a thread per extrusion which will then itself dynamically launch a thread for
eachcell. Theresultssectiondiscussestheperformanceofbothstrategies.
10 Calculating the signed distance field
WeallocatespaceintheGPUglobalmemoryforthe3Ddomainasarowmajor1Dfloat
pointer. Bystoringthephysicallimits andwidth, heightand depthinformation, we can
findthex,yandzcoordinatesofeachcellfromitsoffsetinthepointer.
TheSDFcalculation kernelfirstchecksifacellcentreiswithintheextrusioninques-
tion by performing a half plane testagainst the sides of the polyhedron for prisms, or a
discriminanttestforconesandhemispheres. Toavoidmachineepsilonerrorsandissues
with testing discrete grid positions against continuum planes, we compare the results
against small values from 10−4∆x to 10−3∆x where ∆x is the length of a cell in one di-
mension. Ifthepointiswithintheextrusion,wecalculatethedistancetothefeature.
For a face with normal n and a point p on its surface, the distance to point c can be
foundbyn·pc. Iftheabsolutevalue issmaller than auserdefinedmaximum, andifthe
previousmagnitudeatthatcellcentreislarger,wewritetheresulttoglobalmemorywith
theappropriatesigndependingontheextrusion. Foredgeextrusions,thedistanceisthe
distancetoalineandforavertex,itisthedistancebetweentwopointsin3D.
11 Results
We present the results and timings of a number of test cases. The code was run on an
Nvidia Tesla K20 card [16] with common STL geometries. We show surface plots with
pseudocolourandisosurfacesoftheSDFandlistthepreprocessinganddistancegenera-
tiontimes.

674 A.Roosing,O.T.StricksonandN.Nikiforakis/Commun.Comput.Phys.,26(2019),pp.654-680
11.1 Accuracy
The produced code was validated against multiple common geometries which feature
complex irregular surfaces as shown in Fig. 13. Fig. 14 shows a zoomed-in region to
illustrate the high resolution of the computational mesh, the SDF being set only in the
immediateregionofthesurface(14(a))andhowtheproducedsurfacematchestheinput
mesh with an expectederror of the order of the cell size (14(b)). (The visualisation soft-
ware interpolates both the SDF values and the surface slices, which makes the image a
closeapproximation,notanexactreproduction.)
(a) Orion[17] (b) StanfordRabbit[18] (c) XYZRGBDragon[18] (d) StanfordLucy[18]
Figure13: SurfaceplotsandSDFslicesoftestgeometries. Narrowbandsigneddistancefieldsweregeneratedfor
complexshapeswithvaryingfeaturecountsontheGPU.Therobustnessandperformanceoftheimplementation
allows for quick preprocessing times in various disciplines.
(a) levelset(grey)withSTLedges(black) (b) level set (solid red) and STL slice
(dashedblack)
Figure14: ResultsofStanfordrabbitearat∆x=0.125,distance=2. (a)showshowtheproducedlevelset(grey)
matchesthemeshlinesoftheSTLtriangles(black). Withsufficientdomainresolutionthecodereproducesthe
sharp discontinuities of the input geometry. (b) shows how the slices of the level set (solid red line) and the
STL (dashed black line) match to within ∆x. Note that thevisualisation software interpolates values and that
thenumerical accuracy is often higher than theimage.

A.Roosing,O.T.StricksonandN.Nikiforakis/Commun.Comput.Phys.,26(2019),pp.654-680 675
(a) Errors when only considering convex (b) Correctresultwhenaddressingsaddle
andconcavevertices points
Figure15: ResultsoftheStanfordrabbitearat∆x=0.03showtheissueswithsaddlepoints. Whenonlyassigning
extrusions to convex and concave vertices, holes are left at saddle points. As no extrusion assigns a correct
distance or sign, other extrusions can bleed into these regions. This can result in pyramid artifacts protruding
fromthesurfacewherenegativeextrusionsareneveroverwritten(a). Atlowerresolutionstheerrorscanappear
as artifacts farther away as the region closer to the surface gets the correct sign from nearby extrusions, but
theendsoftheinteriorextrusionsareleftuncorrected. (b)showsthecorrectSDFwhenassigningextrusionsto
saddle points.
Fig.15(a)illustratestheerrorsproducedbyonlyconsideringconvexandconcavever-
tices at the right ear of the Stanford rabbit geometry [18]. When not addressing saddle
points,gapsareleftintowhichnearbyextrusionsmayextend. Asthesevaluesarenever
overwritten, artifacts may be produced. When a negative extrusion is not overwritten
by a smaller magnitude positive extrusion on the outside of the surface, pyramid like
protrusions are created in the level set. These errors may also appear as farther away
spheres when the values near the surface are covered by neighbouring positive extru-
sions. Fig. 15(b) shows the correctly produced SDF by generating extrusions on both
sidesofsaddleverticesbybuildingaconearoundpositivepointingnormalsandreflect-
ingthemtothenegativepseudonormaldirection.
While hemisphere generation at ruff geometries will produce the correct SDF, it
emerged that it is sufficient to consider a cone extrusion restricted to positive pointing
normals. Though the correctness of this approach is not certain, in all of the test cases,
a cone encompassing just the positive pointing normals produced no gaps. The vol-
ume ofsuchan extrusionis lessthan hemisphereand theworkloadis thereforesmaller.
Fig. 16 shows the SDF for the ruff geometry of Fig. 7(b). The hole left at the convex
vertex is filled by generatinga cone enclosing the positive pointing normals. Following
several attempts, no surface could be found which would lead to an incorrect SDF, al-
thoughit is possible that such a configuration can occur in common geometries. Fig. 17
shows a pathological test case which features a normal pointing almost in the negative

676 A.Roosing,O.T.StricksonandN.Nikiforakis/Commun.Comput.Phys.,26(2019),pp.654-680
Figure 16: A continuous SDF around a ruff geometry. A ruff vertex is classified as convex but the normals
of the faces meeting at it span R3. By considering only the normals which point to the positive side of the
pseudonormal plane, a strictly less than half-space volume can be filled with the distance to the vertex. The
result is a continuoussigned distancefield around thesurface.
(a) Pathologicalsurface (b) Holeleftatvertex (c) SDFincone (d) CorrectSDF
Figure 17: The pathological geometry case features a normal at a vertex which points in almost the opposite
direction to the pseudonormal while the overall geometry is convex. Generating a cone of positive pointing
normals fills thegap left between other extrusions.
pseudonormal direction with the vertex being categorised as convex (17(a)). We show
the hole left from other features (17(b)), the SDF in the cone around positive pointing
normals(17(c))andthecorrectdistancefieldwhenapplyingtheextrusion(17(d)).
11.2 Performance
Table1showsthefeaturecountsandgenerationtimesofinternalgeometrydataforvar-
ious bodies. We list the minimum recorded durations of several runs per shape. This
includes reading in a binary STL file, generating entries on the CPU, copying them to
the GPU where vertices and edges are sorted and combined into unique features. This
timing also includes the construction ofthe extrusionpolyhedraon the GPU. We note a
stable scaling which dependsheavily on the feature count. The timing also dependson
theuniformityoftrianglesizesandtheextentoftheSTLgeometrywhich determinethe
uniquenessofMortoncodesandhowmuchserialisationoccursinfeatureconstruction.
Table 2 shows the number of vertices listed in the input STL file, how many unique
pointstheyarecombinedintoandwhattheproportionofsaddlepointsis. Notallunad-

A.Roosing,O.T.StricksonandN.Nikiforakis/Commun.Comput.Phys.,26(2019),pp.654-680
677
| Table | 1: Internalgeometry |                | generation   | times for different |         | STL files | on K20 card. |
| ----- | ------------------- | -------------- | ------------ | ------------------- | ------- | --------- | ------------ |
|       |                     |                | Geometry     | Faces               | Time(s) |           |              |
|       |                     |                | Orion        | 51,770              | 0.095   |           |              |
|       |                     | StanfordRabbit |              | 69,664              | 0.114   |           |              |
|       |                     | StanfordDragon |              | 100,000             | 0.154   |           |              |
|       |                     | XYZRGBDragon   |              | 721,788             | 0.951   |           |              |
|       |                     |                | StanfordLucy | 2,529,647           | 3.105   |           |              |
|       |                     |                | DrivAer      | 2,854,762           | 3.601   |           |              |
Table 2: The STL file format lists each vertex multiple times, and the resulting software combines them into
uniquepointsfrom which theappropriateextrusions aregenerated. Forcomplex geometries, alargefraction of
vertices are saddle points and need extrusions on both sides of the surface. The number of actual holes and
errorsintheSDFisdifferentdependingonthetargetdomainresolutionandtheconfigurationofthesurrounding
surface.
|                | Geometry     |     | Totalvertices | Unique    | Saddle  |     | Proportion |
| -------------- | ------------ | --- | ------------- | --------- | ------- | --- | ---------- |
|                | Orion        |     | 155,310       | 25,876    | 9,795   |     | 37.8%      |
| StanfordRabbit |              |     | 208,992       | 34,834    | 17,624  |     | 50.5%      |
| StanfordDragon |              |     | 300,000       | 50,000    | 26,431  |     | 52.8%      |
| XYZRGBDragon   |              |     | 2,165,364     | 360,894   | 192,882 |     | 53.4%      |
|                | StanfordLucy |     | 7,589,232     | 1,264,847 | 620,974 |     | 49.1%      |
|                | DrivAer      |     | 8,564,286     | 1,427,345 | 595,337 |     | 41.7%      |
dressed saddle vertices lead to visible errors in the SDF as surrounding extrusions may
combine into watertight surfaces and depending on the resolution of the target mesh,
the errors may not even be noticeable. However, the resulting SDF will not be accurate
foreveryresolutionandthelikelihoodofdisruptiveerrorsincreaseswiththenumberof
saddle points. For the complex test surfaces, the number of saddle points was between
37.8% and53.4%, makingit necessarytohave arobuststrategytodealwith highcurva-
turevertices.
Table 3 shows the time spent on generating the SDF for an Nvidia K20 card using
dynamic parallelism. They list the minimum recorded durations of multiple runs. This
includes kernel launches and tests if cells are within extrusions and writing appropri-
ate values to global memory. The results show short generation time for the simpler
testcases but also poorscaling for higher feature counts. Table 4 showsthe times when
looping through bounding volume cells and not using dynamic parallelism. While the
runtimes for simpler test cases are longer than for the parallel approach, as the feature
count increases, the serial approach outperformsthe alternative. This is due to both the
higher launch cost of kernels on the device and a limited queue of active kernels and
threads. The tipping pointin performance is around 105 faces, past which the serial ap-
proach is consistentlybetter. An optimal implementation would thenfind a balance be-
tweenmaintaining themaximum amountofactive parallel calculation andmakingsure

A.Roosing,O.T.StricksonandN.Nikiforakis/Commun.Comput.Phys.,26(2019),pp.654-680
678
Table 3: SDF generation times in seconds for test geometries on K20 with dynamic parallelism.
|          |                 | CellSize∆x |       |                   | CellSize∆x |       |                   | CellSize∆x |        |
| -------- | --------------- | ---------- | ----- | ----------------- | ---------- | ----- | ----------------- | ---------- | ------ |
| Distance |                 |            |       | Distance          |            |       | Distance          |            |        |
|          |                 | 0.08       | 0.04  |                   | 0.25       | 0.125 |                   | 0.16       | 0.08   |
|          | 0.4             | 0.174      | 0.327 | 2                 | 0.223      | 0.242 |                   | 2 0.335    | 0.748  |
|          | 0.8             | 0.180      | 0.486 | 5                 | 0.239      | 0.803 |                   | 5 1.090    | 7.192  |
|          | (a)Oricon       |            |       | (b)StanfordRabbit |            |       | (c)StanfordDragon |            |        |
|          |                 | CellSize∆x |       |                   | CellSize∆x |       |                   | CellSize∆x |        |
| Distance |                 |            |       | Distance          |            |       | Distance          |            |        |
|          |                 | 0.53       | 0.26  |                   | 4          | 2     |                   | 11.4e-3    | 5.7e-3 |
|          | 5               | 2.214      | 2.266 | 20                | 7.688      | 7.700 | 0.06              | 8.535      | 8.513  |
|          | 10              | 2.258      | 5.277 | 40                | 7.721      | 7.741 | 0.12              | 8.490      | 8.840  |
|          | (d)XYZRGBDragon |            |       | (e)StanfordLucy   |            |       |                   | (f)DrivAer |        |
Table 4: SDF generation times in seconds for test geometries on K20 without dynamic parallelism.
|          |                 | CellSize∆x |       |                   | CellSize∆x |       |                   | CellSize∆x |        |
| -------- | --------------- | ---------- | ----- | ----------------- | ---------- | ----- | ----------------- | ---------- | ------ |
| Distance |                 |            |       | Distance          |            |       | Distance          |            |        |
|          |                 | 0.08       | 0.04  |                   | 0.25       | 0.125 |                   | 0.16       | 0.08   |
|          | 0.4             | 0.234      | 1.745 | 2                 | 0.072      | 0.518 |                   | 2 0.123    | 0.866  |
|          | 0.8             | 0.318      | 2.415 | 5                 | 0.257      | 1.941 |                   | 5 1.165    | 9.160  |
|          | (a)Oricon       |            |       | (b)StanfordRabbit |            |       | (c)StanfordDragon |            |        |
|          |                 | CellSize∆x |       |                   | CellSize∆x |       |                   | CellSize∆x |        |
| Distance |                 |            |       | Distance          |            |       | Distance          |            |        |
|          |                 | 0.53       | 0.26  |                   | 4          | 2     |                   | 11.4e-3    | 5.7e-3 |
|          | 5               | 0.143      | 0.702 | 20                | 0.071      | 0.316 | 0.06              | 0.078      | 0.339  |
|          | 10              | 0.591      | 3.972 | 40                | 1.395      | 1.548 | 0.12              | 0.277      | 1.616  |
|          | (d)XYZRGBDragon |            |       | (e)StanfordLucy   |            |       |                   | (f)DrivAer |        |
thehardware queueis notoversubscribedby doing serialtraversal ofbondingvolumes
thatmayotherwisewaittoolongfordevicesidelaunch.
BothTables3and4showhowtheruntimedependsonthecellsizeofthedomainand
themaximumdistanceoftheSDF.Thesevariablesarethemainmeasuresofworkloadfor
single boundingvolumes. As thenumber ofcells in thevolumes increases, more points
needtobetestedforinclusionintheextrusionswhichmeansincreasedkernellaunchesor
longer cell looping and potentially more conflicts in the atomic write to global memory.
For the purposes of embedded mesh calculations a distance of only a couple of cells
is needed to produce an accurate surface description which we can demonstrate short
runtimesfor.
11.3 Limitations
Whilethecurrentimplementationintroducessomeimprovements,therestillremainlim-
itations to the underlying algorithm. The CSC algorithm assumes a correct orientable

A.Roosing,O.T.StricksonandN.Nikiforakis/Commun.Comput.Phys.,26(2019),pp.654-680 679
surface, which means that there can be no flipped faces or gaps betweenfaces. It is still
possible to produce a correct SDF of a non-closed surface when clipping it to a smaller
computationalmeshwhereeverythinginthedomainiseitherononeortheothersideof
thesurface. Theproducedapproachonlycreatesanarrowbandaroundthesurface,lead-
ingtoasecondaryzerocrossingbetweenthenegativelimitoftheSDFandtheinteriorof
the surface beyond the maximum distance. This can be easily fixed by sweeping along
each of the coordinate axes and filling in unset values in the interior of the geometry.
Thegeometrygenerationmaybeslowforlargegeometrieswithwidelyvaryingtriangle
sizes. In such domains, many smaller triangles can be assigned the same Morton code,
leading to greater serialisation of the feature construction and longer generation times.
Thiscanbeaddressedbysubdividingtheinputordistributingitacrossmultiplecards.
12 Conclusion
OurworkfocusedondescribingembeddedgeometriesinCFDsimulations. Theseoften
featurerelativelyhighresolutionsanddomainsthatextendfarbeyondtheobjectsurface.
ThereisaneedforquicklygeneratingtheSDFofcomplexgeometriesin limitedregions
of space, which still comprise a high number of small cells. The produced implemen-
tation allows for quick organisation and construction of internal geometry information,
workschedulingandgeneratingasigneddistancefieldnearobjectboundaries.
Theoriginal CSCalgorithmhasbeenadjustedtoinclude angle weightedvertexnor-
malsandfixesforsaddlepointsasseeninliterature. Wehavealsopresentedadiscussion
onproblemsoftheoriginalalgorithmathighcurvatureverticesandafixforthesecases.
Adiscussiononthenatureoftheextrusionshasshownthattherearenoareasleftuncov-
eredbytheunionofextrusionsandthatsignconflictsdonotleadtoambiguity. Though
ahemisphereextrusionisthemostcertainwaytoensureacorrectSDFathighcurvature
vertices,inpractice,aconeofpositivepointingnormalsissufficient.
By using a set of common 3D geometrytest cases, we have shown the robustnessof
the algorithm and demonstrated the performance of both the geometry preparation as
wellastheSDFgenerationforarangeoffeaturecountsanddomainresolutions. Likethe
originalimplementation ofthealgorithm, theperformancescales withthefeaturecount
ofthetriangulatedsurfaceandthenumberofcellswithintheboundingvolumes.
We have presented a high performance generation of the necessary geometric data
and theschedulingofworkon GPUs. The resultingimplementation offers arobustand
fastwayofgenerating3DsigneddistancefieldsinhighresolutionCartesiangrids.
References
[1] Bærentzen,J.A.andAanæs,H.,2005.Signeddistancecomputationusingtheangleweighted
pseudonormal. IEEE Transactionson Visualizationand Computer Graphics, 11(3),pp.243-
253.

680 A.Roosing,O.T.StricksonandN.Nikiforakis/Commun.Comput.Phys.,26(2019),pp.654-680
[2] Bridson, R., Marino, S. and Fedkiw, R., 2005, July. Simulation of clothing with folds and
wrinkles.InACMSIGGRAPH2005Courses(p.3).ACM.
[3] Fedkiw, R.P., Aslam, T., Merriman, B. and Osher, S., 1999. A non-oscillatory Eulerian ap-
proach to interfaces in multimaterial flows (the ghost fluid method). Journal of computa-
tionalphysics,152(2),pp.457-492.
[4] Janßen,C.F.,Koliha, N.andRung, T.,2015.Afastandrigorouslyparallelsurfacevoxeliza-
tion technique for GPU-accelerated CFD simulations. Communications in Computational
Physics,17(5),pp.1246-1270.
[5] Mauch, S., 2000. A fast algorithm for computing the closest point and distance transform.
http://www.acm.caltech.edu/seanm/software/cpt/cpt.pdf.
[6] Nickolls,J.,Buck,I.,Garland,M.andSkadron,K.,2008,August.Scalableparallelprogram-
mingwithCUDA.InACMSIGGRAPH2008classes(p.16).ACM.
[7] Park,T.,Lee,S.H.,Kim, J.H.andKim, C.H.,2010,June.CUDA-basedsigneddistancefield
calculation for adaptive grids. In Computer and Information Technology (CIT), 2010IEEE
10thInternationalConferenceon(pp.1202-1206).IEEE.
[8] Peikert,R.andSigg,C.,2005.OptimizedboundingpolyhedraforGPU-baseddistancetrans-
form.InProceedingsofDagstuhlSeminar023231onScientificVisualization.
[9] Pharr,M.,Jakob,W.andHumphreys,G.,2016.Physicallybasedrendering: Fromtheoryto
implementation.MorganKaufmann.
[10] Sigg,C.,Peikert,R.andGross,M.,2003,October.Signeddistancetransformusinggraphics
hardware. In Proceedings of the 14th IEEE Visualization 2003 (VIS’03) (p. 12). IEEE Com-
puterSociety.
[11] Sud,A.,Otaduy,M.A.andManocha,D.,2004,September.DiFi: Fast3Ddistancefieldcom-
putation using graphics hardware. In Computer Graphics Forum (Vol. 23, No. 3, pp. 557-
566).BlackwellPublishing,Inc.
[12] Tang, X., Pattnaik, A., Jiang, H., Kayiran, O., Jog, A., Pai, S., Ibrahim, M., Kandemir, M.T.
andDas,C.R.,2017,February.ControlledKernelLaunchfordynamicparallelisminGPUs.
InHighPerformanceComputerArchitecture(HPCA),2017IEEEInternationalSymposium
on(pp.649-660).IEEE.
[13] Wang, J. and Yalamanchili, S., 2014, October. Characterization and analysis of dynamic
parallelisminunstructuredGPUapplications.InWorkloadCharacterization(IISWC),2014
IEEEInternationalSymposiumon(pp.51-60).IEEE.
[14] Wong,K.V.andHernandez,A.,2012.Areviewofadditivemanufacturing.ISRNMechanical
Engineering,2012.
[15] Drivaer model. https://www.aer.mw.tum.de/en/research-groups/automotive/drivaer/.
Retrieved12.2017.
[16] Nvidia Tesla K20. http://www.nvidia.co.uk/content/tesla/pdf/NVIDIA-Tesla-Kepler-
Family-Datasheet.pdf.Retrieved12.2017.
[17] Orion capsule, nasa 3d resources. https://nasa3d.arc.nasa.gov/detail/orion-capsule. Re-
trieved12.2017.
[18] TheStanford3Dscanning repository.http://graphics.stanford.edu/data/3Dscanrep/.Re-
trieved12.2017.
[19] Thrustlibrary.http://docs.nvidia.com/cuda/thrust/.Retrieved12.2017.
[20] SeanMauch.stdlib.https://bitbucket.org/seanmauch/stlib/src/.Retrieved06.2016.