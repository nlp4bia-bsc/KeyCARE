%%time

#sample texts
text1 = "Paciente varón de 35 años con tumoración en polo superior de teste derecho hallada de manera casual durante una autoexploración, motivo por el cual acude a consulta de urología donde se realiza exploración física, apreciando masa de 1cm aproximado de diámetro dependiente de epidídimo, y ecografía testicular, que se informa como lesión nodular sólida en cabeza de epidídimo derecho. Se realiza RMN. Confirmando masa nodular, siendo el tumor adenomatoide de epidídimo la primera posibilidad diagnóstica. Se decide, en los dos casos, resección quirúrgica de tumoración nodular en cola epidídimo derecho, sin realización de orquiectomía posterior. En ambos casos se realizó examen anátomopatológico de la pieza quirúrgica. Hallazgos histológicos macroscópicos: formación nodular de 1,5 cms (caso1) y 1,2 cms (caso 2) de consistencia firme, coloración blanquecina y bien delimitada. Microscópicamente se observa proliferación tumoral constituida por estructuras tubulares en las que la celularidad muestra núcleos redondeados y elongados sin atipia citológica y que ocasionalmente muestra citoplasmas vacuolados, todo ello compatible con tumor adenomatoide de epidídimo."
text2 = "Dos recién nacidos, varón y hembra de una misma madre y fallecidos a los 10 y 45 minutos de vida respectivamente a los que se les realizó examen necrópsico. El primero de los cadáveres, correspondiente a la hembra, fue remitido con el juicio clínico de insuficiencia respiratoria grave con sospecha de Síndrome de Potter con la constatación de oligoamnios severo; nació mediante cesárea urgente por presentación de nalgas y el test de Apgar fue 1/3/7; minutos más tarde falleció. El examen externo permitió observar una tonalidad subcianótica, facies triangular con hendiduras parpebrales mongoloides, micrognatia, raiz nasal ancha y occipucio prominente. El abdomen, globuloso, duro y ligeramente abollonado permitía la palpación de dos grandes masas ocupando ambas fosas renales y hemiabdomenes. A la apertura de cavidades destacaba la presencia de dos grandes masas renales de 10 x 8 x 5,5 cm y 12 x 8 x 6 cm con pesos de 190 y 235 gr respectivamente. Si bien se podía discernir la silueta renal, la superficie, abollonada, presentaba numerosas formaciones quísticas de contenido seroso; al corte dichos quistes mostraban un tamaño heterogéneo siendo mayores los situados a nivel cortical, dando al riñón un aspecto de esponja. Los pulmones derecho e izquierdo pesaban 17 y 15 gr (peso habitual del conjunto de 49 gr) mostrando una tonalidad rojiza uniforme; ambos se encontraban comprimidos como consecuencia de la elevación diafragmática condicionada por el gran tamaño de los riñones. El resto de los órganos no mostraba alteraciones macroscópicas significativas salvo las alteraciones posicionales derivadas de la compresión renal. En el segundo de los cadáveres, el correspondiente al varón, se observaron cambios morfológicos similares si bien el tamaño exhibido por los riñones era aún mayor, con pesos de 300 y 310 gr. El resto de las vísceras abdominales estaban comprimidas contra el diafragma. En ambos casos se realizó un estudio histológico detallado, centrado especialmente en los riñones en los que se demostraron múltiples quistes de distintos tamaños con morfología sacular a nivel cortical. Dichos quistes ocupaban la mayor parte del parénquima corticomedular si bien las zonas conservadas no mostraban alteraciones significativas salvo inmadurez focal. Dichos quistes estaban tapizados por un epitelio simple que variaba desde plano o cúbico. Los quistes medulares, de menor tamaño y más redondeados estaban tapizados por un epitelio de predominio cúbico. Después de las renales, las alteraciones más llamativas se encontraban en el hígado donde se observaron proliferación y dilatación, incluso quística, de los ductos biliares a nivel de los espacios porta. Con tales hallazgos se emitió en ambos casos el diagnóstico de enfermedad poliquística renal autosómica recesiva infantil."
text3 = "Paciente de 64 años, alérgico a penicilina y con recambio valvular aórtico por endocarditis que consultó por aparición de masa peneana de crecimiento progresivo en las últimas semanas. A la exploración física destacaba una formación excrecente y abigarrada en glande, que deformaba meato, con áreas ulceradas cubiertas de fibrina. Se palpaban adenopatías fijas y duras en ambas regiones inguinales. La radiografía de tórax y el TAC abdomino-pélvico confirmaron la presencia de adenopatías pulmonares e inguinales de gran tamaño. Con el diagnóstico de neoplasia de pene, se practicó penectomía parcial con margen de seguridad. La anatomía patológica demostró que se trataba de un sarcoma pleomórfico de pene con diferenciación osteosarcomatosa y márgenes libres de afectación. Se decidió tratamiento con dos líneas de quimioterapia consistente en adriamicina e ifosfamida pero no hubo respuesta. Ingresó de nuevo con recidiva local sangrante de gran tamaño y crecimiento rápido que provocaba obstrucción de meato con insuficiencia renal aguda. Se colocó sonda de cistostomía y se instauró tratamiento con sueroterapia, mejorando la función renal, pero con empeoramiento progresivo del estado general hasta que falleció a los 6 meses del diagnóstico."
text4 = "Mujer de 28 años sin antecedentes de interés que consultó por síndrome miccional con polaquiuria de predominio diurno y cierto grado de urgencia sin escapes urinario. El urocultivo resultó negativo por lo que se indicó tratamiento con anticolinérgicos. Ante la falta de respuesta al tratamiento, se realizó cistografía que fué normal y ecografía renovesical en la que se apreciaban imágenes quísticas parapiélicas, algunas de ellas con tabiques internos y vejiga sin lesiones. Con el fin de precisar la naturaleza de dichos quistes se solicitó TAC-abdominal, que informaba de gran quiste parapiélico en riñón derecho sin repercusión sobre la vía y una masa hipodensa suprarrenal derecha. La resonancia magnética demostró normalidad de la glándula suprarrenal y una lesión quística lobulada conteniendo numerosos septos en su interior que rodeaba al riñón derecho; en la celda renal izquierda existía una lesión de características similares pero de menor tamaño. Los hallazgos eran compatibles con linfangioma renal bilateral. Tras tres años de seguimiento la paciente continua con leve síntomatologia miccional en tratamiento, pero no ha presentado síntomas derivados de su lesión renal."
text5 = "Varón de 68 años, con antecedentes de hemorragia digestiva alta por aspirina y accidente isquémico transitorio a tratamiento crónico con trifusal (300 mg cada 12 horas), que acudió al Servicio de Urgencias del Hospital San Agustín (Avilés, Asturias), en mayo de 2006, por dolor en hemiabdomen izquierdo, intenso, continuo, de instauración súbita y acompañado de cortejo vegetativo. A la exploración presentaba una tensión arterial de 210/120 mm Hg, una frecuencia cardíaca de 80 por minuto, y dolor en fosa ilíaca izquierda, acentuado con la palpación. El hemograma (hemoglobina: 13 g/dL, plaquetas: 249.000), el estudio de coagulación, la bioquímica elemental de sangre, el sistemático de orina, el electrocardiograma y la radiografía simple de tórax eran normales. En la tomografía computarizada de abdomen se objetivó un extenso hematoma, de 12 cm de diámetro máximo, en la celda renal izquierda, sin líquido libre intraperitoneal; la suprarrenal izquierda quedaba englobada y no se podía identificar, y la derecha no presentaba alteraciones. La HTA no se llegó a controlar en Urgencias, a pesar del tratamiento con analgésicos, con antagonistas del calcio y con inhibidores de la enzima convertidora de la angiotensina II, por lo que el paciente, que mantenía cifras tensionales de 240/160 mm Hg, pasó a la Unidad de Cuidados Intensivos, para tratamiento intravenoso con nitroprusiato y labetalol. En las 24 horas siguientes se yuguló la crisis hipertensiva, y se comprobó que la hemoglobina y el hematocrito permanecían estables. Con la sospecha diagnóstica de rotura no traumática de un feocromocitoma pre-existente, se determinaron metanefrinas plasmáticas, que fueron normales, y catecolaminas y metanefrinas urinarias. En la orina de 24 horas del día siguiente al ingreso se obtuvieron los siguientes resultados: adrenalina: 65,1 mcg (valores normales -VN: 1,7-22,5), noradrenalina: 151,1 mcg (VN: 12,1-85,5), metanefrina: 853,5 mcg (VN: 74-297) y normetanefrina: 1396,6 mcg (VN: 105-354). A los 10 días, todavía ingresado el paciente, las cifras urinarias se habían normalizado por completo de modo espontáneo. Respecto al hematoma, en julio de 2006 no se había reabsorbido y persistía una imagen pseudoquística en la zona suprarrenal izquierda. En septiembre de 2006 se practicó una suprarrenalectomía unilateral, y el estudio histológico mostró una masa encapsulada de 6 x 5 cm, con necrosis hemorrágica extensa y algunas células corticales sin atipias."

extractor = TermExtractor(extraction_methods=["yake"])
extractor(text1)
print(len(extractor.keywords))
extractor(text2)
print(len(extractor.keywords))
extractor(text3)
print(len(extractor.keywords))
extractor(text4)
print(len(extractor.keywords))
extractor(text5)
print(len(extractor.keywords))