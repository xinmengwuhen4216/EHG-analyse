import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams.update({'font.sans-serif': 'Times New Roman', 'xtick.labelsize': 16, 'ytick.labelsize': 16})
plt.rcParams.update({'font.size': 13})     #设置图例字体大小

num=200
def list_generator(mean, dis, number):  # 封装一下这个函数，用来后面生成数据
    return np.random.normal(mean, dis * dis, number)  # normal分布，输入的参数是均值、标准差以及生成的数量

Graph_sensitivity= [0.7313390313390313, 0.7978632478632478, 0.8169515669515668, 0.8736467236467236, 0.7974358974358975, 0.8052706552706553, 0.8203703703703704, 0.8323361823361823, 0.8283475783475783, 0.7474358974358976, 0.8703703703703705, 0.8172364672364673, 0.8173789173789174, 0.8196581196581196, 0.8092592592592591, 0.8547008547008546, 0.8133903133903134, 0.8280626780626781, 0.8121082621082621, 0.7072649572649572, 0.8313390313390313, 0.812962962962963, 0.8482905982905983, 0.7623931623931625, 0.8202279202279202, 0.8662393162393162, 0.734045584045584, 0.8752136752136751, 0.8242165242165242, 0.8247863247863249, 0.7514245014245015, 0.8435897435897436, 0.8357549857549857, 0.8662393162393162, 0.726068376068376, 0.8173789173789174, 0.7633903133903134, 0.7740740740740741, 0.8324786324786325, 0.8049857549857551, 0.8353276353276353, 0.8239316239316239, 0.844017094017094, 0.7988603988603987, 0.8210826210826211, 0.8235042735042735, 0.8512820512820513, 0.8512820512820513, 0.7867521367521368, 0.8585470085470085, 0.8086894586894587, 0.8243589743589743, 0.8138176638176638, 0.8049857549857549, 0.8256410256410257, 0.8330484330484331, 0.8511396011396011, 0.7700854700854701, 0.8394586894586895, 0.8739316239316238, 0.871082621082621, 0.7831908831908831, 0.8816239316239315, 0.8245014245014245, 0.8437321937321937, 0.7863247863247863, 0.7857549857549857, 0.8668091168091168, 0.8237891737891738, 0.7814814814814814, 0.8132478632478632, 0.8396011396011396, 0.801139601139601, 0.7632478632478632, 0.7742165242165242, 0.8179487179487179, 0.7947293447293446, 0.8132478632478632, 0.8205128205128205, 0.7589743589743589, 0.758974358974359, 0.75997150997151, 0.7605413105413106, 0.832051282051282, 0.766951566951567, 0.8287749287749288, 0.847008547008547, 0.8391737891737892, 0.7514245014245013, 0.8156695156695157, 0.8354700854700855, 0.8207977207977208, 0.7933048433048433, 0.847008547008547, 0.8015669515669515, 0.8622507122507124, 0.8206552706552707, 0.816951566951567, 0.8012820512820513, 0.8012820512820513, 0.8200854700854702, 0.8128205128205128, 0.8163817663817664, 0.7470085470085469, 0.8356125356125356, 0.8088319088319087, 0.7819088319088319, 0.8247863247863247, 0.7451566951566951, 0.836039886039886, 0.739031339031339, 0.7951566951566952, 0.7368945868945869, 0.8319088319088319, 0.7972934472934472, 0.7534188034188034, 0.751994301994302, 0.8132478632478632, 0.8407407407407408, 0.8472934472934475, 0.8173789173789174, 0.8548433048433048, 0.8219373219373219, 0.8203703703703704, 0.8518518518518519, 0.8514245014245013, 0.8467236467236468, 0.8437321937321938, 0.7660968660968661, 0.8547008547008546, 0.8282051282051281, 0.8132478632478632, 0.815954415954416, 0.8391737891737894, 0.8200854700854702, 0.8166666666666667, 0.8236467236467236, 0.8398860398860398, 0.8173789173789174, 0.8284900284900285, 0.7220797720797721, 0.8659544159544159, 0.844017094017094, 0.784045584045584, 0.8514245014245013, 0.7552706552706553, 0.8095441595441596, 0.81994301994302, 0.8084045584045583, 0.8024216524216523, 0.7706552706552707, 0.8321937321937322, 0.8319088319088319, 0.8283475783475783, 0.8477207977207979, 0.8857549857549858, 0.8052706552706551, 0.8398860398860399, 0.839031339031339, 0.8471509971509972, 0.8132478632478632, 0.78988603988604, 0.8350427350427351, 0.8371794871794872, 0.7482905982905983, 0.8504273504273504, 0.8319088319088319, 0.7742165242165242, 0.8316239316239316, 0.8477207977207979, 0.8206552706552707, 0.78988603988604, 0.8014245014245013, 0.8437321937321938, 0.8289173789173787, 0.8552706552706552, 0.8313390313390313, 0.8313390313390313, 0.8123931623931625, 0.8470085470085469, 0.8323361823361823, 0.84002849002849, 0.8478632478632478, 0.8626780626780628, 0.8173789173789172, 0.6834757834757834, 0.7826210826210825, 0.8049857549857549, 0.8356125356125356, 0.758974358974359, 0.8623931623931625, 0.8393162393162392, 0.8551282051282051, 0.7508547008547007, 0.8629629629629629, 0.836039886039886, 0.7937321937321938, 0.8128205128205128, 0.8366096866096866, 0.8323361823361823]
Graph_specificity= [0.8135094339622642, 0.8095094339622643, 0.752, 0.7857358490566038, 0.782188679245283, 0.7746415094339623, 0.7826415094339623, 0.8248301886792454, 0.7744150943396226, 0.7666415094339623, 0.7895094339622641, 0.789056603773585, 0.7666415094339623, 0.7739622641509435, 0.798188679245283, 0.8121509433962265, 0.794188679245283, 0.7746415094339621, 0.7704150943396225, 0.8250566037735849, 0.7739622641509435, 0.7564528301886793, 0.7868679245283019, 0.7979622641509433, 0.7824150943396226, 0.8246037735849058, 0.7708679245283019, 0.8088301886792454, 0.7819622641509435, 0.7859622641509434, 0.793509433962264, 0.8008301886792454, 0.7975094339622641, 0.7828679245283018, 0.7817358490566038, 0.7655094339622643, 0.7819622641509433, 0.8027924528301886, 0.7737358490566038, 0.813056603773585, 0.8319245283018869, 0.7779622641509434, 0.7828679245283019, 0.7784150943396227, 0.7788679245283019, 0.7637735849056604, 0.7826415094339623, 0.800377358490566, 0.7972830188679245, 0.8086037735849058, 0.7744150943396226, 0.7590943396226415, 0.7937358490566038, 0.766867924528302, 0.782188679245283, 0.750867924528302, 0.7739622641509435, 0.824377358490566, 0.8026415094339623, 0.7797735849056604, 0.764, 0.7755471698113207, 0.8015094339622643, 0.8095094339622643, 0.8012830188679245, 0.7750943396226415, 0.8057358490566038, 0.7790943396226415, 0.7972830188679245, 0.798188679245283, 0.7569056603773584, 0.7988679245283019, 0.7819622641509435, 0.7868679245283019, 0.770188679245283, 0.7870943396226415, 0.7668679245283018, 0.7788679245283019, 0.7790943396226415, 0.7977358490566038, 0.7788679245283019, 0.8017358490566038, 0.770188679245283, 0.794188679245283, 0.794188679245283, 0.7870943396226415, 0.813056603773585, 0.7866415094339623, 0.7642264150943396, 0.7692830188679245, 0.7977358490566038, 0.7972830188679245, 0.732, 0.8026415094339624, 0.8048301886792453, 0.7784150943396227, 0.8019622641509434, 0.7706415094339624, 0.774867924528302, 0.7790943396226415, 0.8055094339622644, 0.7750943396226415, 0.7635471698113209, 0.7830943396226415, 0.7944150943396225, 0.794188679245283, 0.7833207547169811, 0.7977358490566038, 0.7786415094339623, 0.802188679245283, 0.7750943396226415, 0.7757735849056604, 0.7859622641509434, 0.8248301886792454, 0.7817358490566038, 0.7972830188679246, 0.7946415094339623, 0.7859622641509434, 0.798188679245283, 0.765056603773585, 0.8017358490566038, 0.7826415094339623, 0.7857358490566038, 0.8088301886792454, 0.7786415094339624, 0.8126037735849059, 0.7870943396226415, 0.7704150943396225, 0.8019622641509434, 0.8128301886792453, 0.7864150943396225, 0.797056603773585, 0.8119245283018868, 0.7670943396226415, 0.785509433962264, 0.7795471698113208, 0.812377358490566, 0.7706415094339624, 0.8095094339622643, 0.7715471698113208, 0.7510943396226415, 0.7670943396226415, 0.7677735849056604, 0.7633207547169812, 0.7790943396226415, 0.7977358490566038, 0.7868679245283019, 0.7906415094339623, 0.802188679245283, 0.7824150943396226, 0.7717735849056604, 0.8024150943396225, 0.7595471698113208, 0.7830943396226415, 0.8135094339622642, 0.782188679245283, 0.7972830188679245, 0.7710943396226415, 0.7795471698113207, 0.7939622641509434, 0.7777358490566039, 0.782188679245283, 0.7675471698113208, 0.7950943396226415, 0.8008301886792454, 0.7666415094339623, 0.774867924528302, 0.783320754716981, 0.7939622641509434, 0.8128301886792453, 0.7790943396226415, 0.768, 0.821056603773585, 0.7897358490566038, 0.8024150943396228, 0.7710943396226415, 0.7972830188679245, 0.774867924528302, 0.770188679245283, 0.7830943396226415, 0.7950943396226415, 0.809509433962264, 0.7717735849056604, 0.7784150943396225, 0.8090566037735849, 0.8121509433962265, 0.7935094339622643, 0.7666415094339623, 0.8212830188679245, 0.8274716981132075, 0.7864150943396225, 0.7677735849056604, 0.7897358490566038, 0.802188679245283, 0.7899622641509435, 0.782188679245283, 0.7746415094339623, 0.829056603773585, 0.8090566037735849, 0.7750943396226415]
Graph_Area= [0.8458272321668548, 0.8637027361178304, 0.8546117292909747, 0.8775702843627371, 0.8592263613395689, 0.8673962264150943, 0.8748915766274259, 0.8835324410041393, 0.8556162984464871, 0.8275869483416655, 0.8938236843519862, 0.8661879266784928, 0.867374939525883, 0.858617749825297, 0.883613073160243, 0.8759111971187442, 0.8656106004407892, 0.8740346180723539, 0.8725419555985592, 0.8317041337418696, 0.8727870773531151, 0.8572952749556523, 0.8564651937859484, 0.8535571681986778, 0.8479715099715099, 0.9053919260334355, 0.8279363543514486, 0.8908032037843359, 0.8532835564156318, 0.8793704241251412, 0.8620824598183088, 0.8813921410525184, 0.8831207869698435, 0.8844354136429609, 0.8292190506907489, 0.8641201956673654, 0.8592248562059883, 0.860881363220986, 0.8871326130194055, 0.873977422996291, 0.9050401548137398, 0.8798307799817234, 0.8838528194377251, 0.8524919636617749, 0.8660033327957857, 0.8582843627371929, 0.8827410632693653, 0.900647852496909, 0.8737302585604473, 0.8898091705638874, 0.8631541149276998, 0.8583438155136267, 0.8713155942589905, 0.8423494060097834, 0.8657926140945008, 0.8641223458581948, 0.8775254528839433, 0.8483957426221578, 0.8891177767026823, 0.8961081545987206, 0.8968580336504866, 0.8513997742299629, 0.9068459925818416, 0.883912702252325, 0.8804059560285975, 0.8527436434983606, 0.8524181046067838, 0.8877110143525238, 0.8715763048970597, 0.8649561898618501, 0.8564394990055366, 0.8717081115949041, 0.8527183787561146, 0.8679595764124066, 0.8598981884642262, 0.8741502983389775, 0.8501625544267053, 0.871594366500027, 0.8748828683545664, 0.8619288286835456, 0.82495543729506, 0.863110143525238, 0.8136935978068053, 0.8621125624899211, 0.8480883728430898, 0.8842951136913401, 0.8893426866634414, 0.8925724883083375, 0.8252736655378164, 0.874508842659786, 0.8885541041767457, 0.8798257270332742, 0.854548513680589, 0.8687665430306941, 0.8681743804762672, 0.8872087297747676, 0.885271300327904, 0.8721590066118369, 0.8631534698704509, 0.8579345266892435, 0.876302854378326, 0.8543193033381712, 0.865827984733645, 0.8280083857442347, 0.8763603719830135, 0.863567489114659, 0.8707957856259745, 0.8759530183303769, 0.8334915873783798, 0.8881111648658818, 0.8580968660968662, 0.8623019943019944, 0.8393208622265226, 0.8768614739558135, 0.8726031285276568, 0.8525591571251949, 0.8398640004300381, 0.8541402999516207, 0.8879567811643282, 0.8812590442401763, 0.8877233779497932, 0.8687765414180507, 0.8785878621727677, 0.8729725313121539, 0.8844773423641348, 0.8806179648443798, 0.8867461162178143, 0.8733791323980004, 0.8326058162661936, 0.8860530022039457, 0.8838825995807127, 0.8678250819760253, 0.8905770037090791, 0.8592768908240606, 0.886762457668118, 0.8739933344084289, 0.8788678170187604, 0.8837621888942644, 0.8864965865720584, 0.885222275976993, 0.8188588937268182, 0.9001873891307854, 0.8819190453152718, 0.8607425684029456, 0.8817772402300704, 0.8538101381497608, 0.8723733806375316, 0.8856070526259205, 0.8575551255173897, 0.8464419717249907, 0.8402761920120412, 0.8725113153792398, 0.8674687953555879, 0.8746756974681503, 0.8733932161479331, 0.8928152448529808, 0.8576286620437562, 0.8863352147503092, 0.8776903725205611, 0.8933906359189377, 0.8656018921679298, 0.8502103961726603, 0.8860391334730957, 0.8971028328764177, 0.8452041068644842, 0.881723592968876, 0.8674937375692094, 0.8454601945922701, 0.8760180616029671, 0.8839058216416706, 0.8630444551953987, 0.8516935978068053, 0.8731131537923991, 0.8967128957694996, 0.893636080202118, 0.894221469655432, 0.8843011342256626, 0.8693245175509325, 0.8496255442670536, 0.87458291673386, 0.8992527011772294, 0.8875432994678277, 0.889414395527603, 0.8843340321453528, 0.8820302101811537, 0.8198723861742729, 0.8568855560931032, 0.8586698919529109, 0.8879199053916034, 0.8450652045369026, 0.8845026071063806, 0.8779572112024943, 0.8984207923453205, 0.859565231414288, 0.8874391227221416, 0.8732438853948288, 0.8631887330000538, 0.8828373918185239, 0.8852924797075741, 0.8667578347578347]
Graph_Accuracy= [0.7731318681318682, 0.8034432234432234, 0.7843406593406593, 0.8301648351648352, 0.790091575091575, 0.7900732600732601, 0.8015750915750915, 0.8282417582417582, 0.8014835164835166, 0.7576739926739926, 0.830091575091575, 0.8032967032967033, 0.7918681318681319, 0.7977655677655677, 0.8034615384615386, 0.8339377289377289, 0.8034065934065934, 0.8015384615384615, 0.7920512820512821, 0.7651648351648352, 0.8034615384615383, 0.7844139194139194, 0.8167216117216117, 0.7806776556776557, 0.8015750915750915, 0.8454578754578754, 0.7517399267399266, 0.8414285714285714, 0.8034249084249085, 0.8052747252747254, 0.7729304029304029, 0.8224542124542124, 0.8167948717948719, 0.8244871794871795, 0.7536446886446886, 0.7917765567765569, 0.7728571428571429, 0.78996336996337, 0.8032967032967033, 0.8092124542124542, 0.8340293040293041, 0.8015384615384615, 0.812948717948718, 0.788003663003663, 0.7995787545787546, 0.7940476190476191, 0.8167948717948719, 0.8262271062271063, 0.7918864468864469, 0.8339560439560438, 0.792014652014652, 0.791959706959707, 0.8033150183150184, 0.7862820512820512, 0.8032600732600732, 0.7917765567765567, 0.8129120879120879, 0.7978205128205129, 0.8207326007326007, 0.8264468864468864, 0.8167399267399267, 0.7785897435897436, 0.8416117216117216, 0.8168131868131867, 0.8224725274725275, 0.7805494505494506, 0.795897435897436, 0.8225091575091575, 0.8111172161172162, 0.7902380952380952, 0.7844139194139194, 0.8188095238095239, 0.7919963369963371, 0.7748717948717949, 0.7729120879120879, 0.8014468864468863, 0.7804029304029304, 0.7957875457875457, 0.7996703296703297, 0.7786996336996337, 0.7691758241758242, 0.7804945054945056, 0.7650732600732599, 0.813003663003663, 0.7805860805860807, 0.8072527472527472, 0.8302014652014652, 0.8130769230769233, 0.7577655677655677, 0.7939010989010988, 0.8168498168498168, 0.8091025641025642, 0.7633699633699633, 0.8245604395604396, 0.8033882783882784, 0.8206410256410257, 0.811098901098901, 0.7938461538461539, 0.7882051282051282, 0.7901465201465202, 0.8130586080586081, 0.7939377289377288, 0.79014652014652, 0.7654578754578755, 0.81496336996337, 0.8015934065934065, 0.7825824175824175, 0.811025641025641, 0.7613369963369964, 0.8187362637362637, 0.7578388278388279, 0.7842673992673993, 0.7613919413919413, 0.8282967032967032, 0.7900732600732601, 0.7745787545787546, 0.772948717948718, 0.7995604395604395, 0.8186080586080585, 0.807087912087912, 0.8091025641025642, 0.8187362637362637, 0.8032234432234432, 0.8148901098901099, 0.8147985347985347, 0.831996336996337, 0.8169413919413919, 0.8071794871794873, 0.7845238095238095, 0.8339926739926741, 0.8072710622710624, 0.8052564102564101, 0.8149084249084249, 0.8034981684981686, 0.8034432234432234, 0.7978021978021979, 0.8187545787545787, 0.8052930402930404, 0.8129304029304029, 0.7996336996336997, 0.7364835164835164, 0.8168498168498168, 0.8053296703296704, 0.7726923076923077, 0.8148901098901099, 0.7767765567765567, 0.7977106227106228, 0.8054761904761906, 0.8054945054945055, 0.7918681318681319, 0.771098901098901, 0.8168498168498168, 0.7958241758241759, 0.8053846153846154, 0.8301465201465202, 0.8339194139194139, 0.8015018315018315, 0.8053296703296704, 0.8093223443223444, 0.8206410256410257, 0.7956959706959708, 0.7862637362637364, 0.8016666666666665, 0.8148168498168499, 0.7747252747252747, 0.8092124542124542, 0.8034615384615386, 0.7787728937728937, 0.813040293040293, 0.830091575091575, 0.7996520146520145, 0.7787179487179487, 0.8111172161172162, 0.8167399267399269, 0.8148717948717948, 0.812948717948718, 0.8149450549450549, 0.8035347985347986, 0.791996336996337, 0.8149816849816851, 0.813040293040293, 0.8244139194139194, 0.8091575091575092, 0.8205860805860807, 0.8128937728937728, 0.747985347985348, 0.788113553113553, 0.7862637362637362, 0.8283333333333334, 0.7939010989010988, 0.8244688644688644, 0.8035347985347986, 0.8224725274725275, 0.7745054945054946, 0.8262820512820512, 0.809120879120879, 0.7843589743589743, 0.8206959706959708, 0.8224175824175826, 0.8034249084249085]

Ori_sensitivity= [0.7418803418803419, 0.7896011396011395, 0.8393162393162392, 0.7702279202279202, 0.7349002849002849, 0.7867521367521367, 0.8396011396011396, 0.7262108262108262, 0.7935897435897437, 0.7243589743589743, 0.804131054131054, 0.7507122507122508, 0.7525641025641026, 0.8017094017094017, 0.8091168091168092, 0.7866096866096866, 0.7135327635327636, 0.7974358974358975, 0.7790598290598292, 0.8475783475783476, 0.7787749287749287, 0.7309116809116809, 0.7702279202279202, 0.812962962962963, 0.7943019943019942, 0.7290598290598291, 0.7282051282051282, 0.8319088319088319, 0.7868945868945869, 0.7595441595441595, 0.7306267806267808, 0.8207977207977208, 0.8289173789173789, 0.8283475783475783, 0.6972934472934472, 0.8055555555555556, 0.6898860398860399, 0.750997150997151, 0.8049857549857549, 0.7592592592592593, 0.7062678062678062, 0.8091168091168092, 0.7091168091168092, 0.7182336182336183, 0.741025641025641, 0.7072649572649572, 0.7638176638176638, 0.7323361823361824, 0.7894586894586894, 0.7978632478632479, 0.7108262108262108, 0.8321937321937322, 0.7698005698005698, 0.6272079772079773, 0.685042735042735, 0.7827635327635327, 0.698860398860399, 0.7797720797720797, 0.7364672364672364, 0.7947293447293446, 0.6826210826210826, 0.7789173789173789, 0.7398860398860398, 0.7977207977207977, 0.7700854700854701, 0.80997150997151, 0.7169515669515669, 0.7596866096866096, 0.8206552706552707, 0.7512820512820513, 0.7757834757834757, 0.6975783475783477, 0.7220797720797721, 0.7982905982905983, 0.799002849002849, 0.7028490028490029, 0.6561253561253562, 0.6792022792022792, 0.7663817663817664, 0.7592592592592593, 0.8400284900284902, 0.7854700854700855, 0.8240740740740741, 0.7048433048433047, 0.7905982905982907, 0.7103988603988605, 0.7329059829059829, 0.7564102564102563, 0.8168091168091168, 0.7752136752136753, 0.7636752136752136, 0.7207977207977208, 0.6823361823361824, 0.808974358974359, 0.8146723646723647, 0.7323361823361823, 0.767094017094017, 0.8132478632478632, 0.8437321937321938, 0.766951566951567, 0.7371794871794871, 0.7125356125356126, 0.7085470085470085, 0.7176638176638177, 0.7588319088319089, 0.7643874643874644, 0.8092592592592591, 0.8324786324786325, 0.7282051282051282, 0.8475783475783475, 0.7568376068376069, 0.7896011396011395, 0.836039886039886, 0.7155270655270656, 0.7554131054131055, 0.721937321937322, 0.8428774928774929, 0.7715099715099715, 0.8203703703703704, 0.6827635327635327, 0.8014245014245015, 0.766951566951567, 0.7524216524216524, 0.7803418803418805, 0.7052706552706554, 0.6609686609686609, 0.8135327635327636, 0.6582621082621083, 0.8162393162393162, 0.808974358974359, 0.7839031339031339, 0.737037037037037, 0.7938746438746439, 0.7522792022792022, 0.7631054131054131, 0.7474358974358973, 0.8434472934472934, 0.8095441595441596, 0.7703703703703704, 0.733048433048433, 0.8094017094017094, 0.7706552706552706, 0.7877492877492877, 0.7250712250712251, 0.6594017094017094, 0.7139601139601139, 0.7867521367521368, 0.8049857549857549, 0.7084045584045584, 0.68005698005698, 0.7717948717948718, 0.8472934472934472, 0.7787749287749287, 0.7588319088319089, 0.8012820512820513, 0.7787749287749287, 0.8092592592592591, 0.784045584045584, 0.7145299145299145, 0.7216524216524217, 0.69002849002849, 0.7215099715099715, 0.6968660968660969, 0.8091168091168092, 0.7790598290598292, 0.7821937321937322, 0.736039886039886, 0.8467236467236466, 0.7665242165242165, 0.7625356125356125, 0.7279202279202279, 0.6974358974358974, 0.7940170940170941, 0.7393162393162392, 0.7292022792022792, 0.7931623931623931, 0.7558404558404558, 0.8282051282051283, 0.6864672364672364, 0.7603988603988604, 0.721937321937322, 0.688034188034188, 0.7126780626780628, 0.7168091168091169, 0.798148148148148, 0.7131054131054131, 0.8047008547008547, 0.8005698005698006, 0.7975783475783474, 0.7141025641025641, 0.8178062678062679, 0.6696581196581197, 0.8321937321937323, 0.7522792022792022, 0.7450142450142451, 0.8243589743589744, 0.7719373219373219, 0.7821937321937322, 0.7534188034188034, 0.685042735042735]
Ori_specificity= [0.7644528301886793, 0.7397735849056605, 0.7717735849056604, 0.7413584905660376, 0.7784150943396225, 0.7602264150943395, 0.774867924528302, 0.7764528301886793, 0.7675471698113207, 0.7557735849056604, 0.7371320754716981, 0.7355471698113207, 0.7630943396226415, 0.7540377358490565, 0.7593207547169811, 0.733811320754717, 0.7750943396226415, 0.7786415094339624, 0.782188679245283, 0.766188679245283, 0.7704150943396227, 0.756, 0.7864150943396225, 0.7666415094339623, 0.7710943396226415, 0.748, 0.7406792452830189, 0.774188679245283, 0.7557735849056604, 0.7406792452830189, 0.7562264150943397, 0.76, 0.775320754716981, 0.7449056603773585, 0.748, 0.770188679245283, 0.7750943396226415, 0.7655094339622641, 0.754188679245283, 0.7395471698113208, 0.7830943396226415, 0.7475471698113209, 0.7784150943396227, 0.7409056603773585, 0.7402264150943395, 0.7517735849056605, 0.7442264150943396, 0.7675471698113208, 0.777056603773585, 0.7264905660377358, 0.7249056603773584, 0.7131320754716981, 0.7886037735849057, 0.7908679245283018, 0.7220377358490566, 0.7746415094339621, 0.7569056603773585, 0.744, 0.7666415094339623, 0.7510943396226415, 0.7977358490566038, 0.7673207547169811, 0.742867924528302, 0.748, 0.763320754716981, 0.7677735849056604, 0.7424150943396228, 0.7646792452830189, 0.7666415094339623, 0.736, 0.7187169811320755, 0.7706415094339624, 0.7246792452830189, 0.7739622641509435, 0.7486792452830189, 0.774867924528302, 0.7673207547169811, 0.7637735849056604, 0.8012830188679245, 0.7786415094339623, 0.7517735849056605, 0.7524528301886793, 0.7477735849056604, 0.7470943396226415, 0.7555471698113208, 0.7409056603773585, 0.7710943396226415, 0.748, 0.7722264150943395, 0.748, 0.7402264150943397, 0.7442264150943396, 0.7857358490566039, 0.7642264150943395, 0.7510943396226416, 0.7675471698113208, 0.7713207547169812, 0.7755471698113208, 0.7784150943396225, 0.763320754716981, 0.7717735849056604, 0.7666415094339623, 0.7637735849056604, 0.7866415094339623, 0.7746415094339623, 0.7597735849056603, 0.744679245283019, 0.7677735849056604, 0.7670943396226415, 0.7908679245283019, 0.7710943396226415, 0.7550943396226415, 0.7529056603773585, 0.7584150943396227, 0.7557735849056604, 0.7364528301886792, 0.7515471698113207, 0.7633207547169812, 0.76, 0.7482264150943395, 0.7522264150943399, 0.7579622641509434, 0.7555471698113208, 0.744, 0.7291320754716981, 0.7595471698113206, 0.7482264150943395, 0.7595471698113208, 0.7593207547169811, 0.7515471698113207, 0.7371320754716981, 0.7526792452830188, 0.7357735849056605, 0.7522264150943396, 0.758188679245283, 0.7788679245283019, 0.754867924528302, 0.7602264150943395, 0.7366792452830188, 0.7331320754716982, 0.7466415094339623, 0.7437735849056605, 0.7550943396226415, 0.7595471698113208, 0.7449056603773585, 0.7837735849056604, 0.7895094339622641, 0.7477735849056604, 0.7895094339622641, 0.7433207547169812, 0.7915471698113207, 0.8048301886792453, 0.7666415094339621, 0.7555471698113208, 0.7430943396226415, 0.7524528301886793, 0.7522264150943396, 0.7529056603773585, 0.7557735849056604, 0.7489056603773585, 0.7326792452830189, 0.790188679245283, 0.7788679245283019, 0.7510943396226416, 0.7637735849056604, 0.7584150943396228, 0.7710943396226415, 0.7404528301886792, 0.744, 0.7642264150943396, 0.7630943396226415, 0.7677735849056604, 0.7590943396226415, 0.801509433962264, 0.7748679245283018, 0.7595471698113206, 0.6975849056603773, 0.771320754716981, 0.745811320754717, 0.7670943396226415, 0.7564528301886791, 0.7493584905660378, 0.7049056603773585, 0.7593207547169811, 0.764, 0.7364528301886792, 0.7630943396226415, 0.7579622641509435, 0.7830943396226415, 0.775320754716981, 0.7744150943396226, 0.766867924528302, 0.7741886792452831, 0.7604528301886793, 0.7517735849056605, 0.7628679245283019, 0.7673207547169811, 0.7286792452830188, 0.779320754716981, 0.7817358490566038]
Ori_Area= [0.8346634413804225, 0.8257889587700907, 0.8517164973391388, 0.8155798527119282, 0.8070989625329249, 0.8351018652905445, 0.8620441864215449, 0.8198756114605171, 0.8508306187174111, 0.8113856904800301, 0.8434670752029241, 0.810044186421545, 0.8201241197656293, 0.8573277428371767, 0.8481492232435629, 0.8008282535074989, 0.8423084448744825, 0.8480537547707359, 0.8321000913831103, 0.8742563027468687, 0.8365367951405688, 0.8065637800354782, 0.8271848626565609, 0.8426511852926947, 0.8358895877009085, 0.8004434768585712, 0.7722177068214804, 0.8606307584798151, 0.8131389560823523, 0.8175630812234586, 0.8191605655001881, 0.8345261516959631, 0.8637849809170565, 0.8467863247863248, 0.7768458850723002, 0.8328619039939795, 0.7664483147879375, 0.8165639950545611, 0.8442052357146697, 0.8046034510562812, 0.8087126807504166, 0.8433009729613504, 0.8096286620437564, 0.8041568564210074, 0.8201039617266032, 0.7869472665699081, 0.8159127022523249, 0.8136548943718755, 0.8439164650862765, 0.8405204536902652, 0.771363113476321, 0.8394509487717035, 0.8394409503843466, 0.7923938074504113, 0.753483416653228, 0.8528635166371016, 0.7699377519754877, 0.832026662366285, 0.7901383647798742, 0.8300179540934257, 0.8071465892597969, 0.839601462129764, 0.7993341934096652, 0.8491805622749018, 0.8496098478739988, 0.850715045960329, 0.8064068161049294, 0.8256179110896091, 0.834236520991238, 0.788604526151696, 0.8133167768639467, 0.7825885072300166, 0.8058523356447885, 0.8461767456861796, 0.8274665376552169, 0.8037754125678653, 0.8021468580336505, 0.7824698166962317, 0.8246242003977853, 0.8167808417997098, 0.8519535558780843, 0.8186344138042252, 0.8472633446218352, 0.7906975219050689, 0.8224050959522659, 0.7983371499220556, 0.8088762027629952, 0.8228020211793796, 0.86088093318282, 0.8194252539912918, 0.8267375154544965, 0.789750255335161, 0.7860428963070473, 0.8283924098263722, 0.8317611137988496, 0.8003139278610977, 0.8119328065365801, 0.842560769768317, 0.8645118529269471, 0.8268492178680859, 0.8066253830027416, 0.8170029565123904, 0.8100694511637908, 0.8221165403429553, 0.8319203891845403, 0.8310163952050745, 0.8388600763317745, 0.8565635650163952, 0.8059930118798043, 0.856258882975864, 0.837163038219642, 0.8138832446379617, 0.8661001988926518, 0.8109860775143793, 0.8081917970219857, 0.779058001397624, 0.8573549427511692, 0.8291413212922647, 0.8489122184593884, 0.7517187550395097, 0.8312184056334999, 0.8189034026769875, 0.8124122990915442, 0.8289658657205827, 0.7866674192334571, 0.778936085577595, 0.8366508627640703, 0.7649032951674462, 0.852223297317637, 0.8382482395312583, 0.8243494060097835, 0.805422136214589, 0.8026314035370639, 0.8078264796000646, 0.8261504058485191, 0.8323845616298445, 0.8469188840509595, 0.8305691555125518, 0.8247076278019675, 0.7941878191689512, 0.8459978498091705, 0.8139416223189808, 0.8233835402892007, 0.8251379884964791, 0.7564287480513896, 0.8060911680911682, 0.8418975434069772, 0.845784013331183, 0.8234694404128365, 0.7728484653012955, 0.8336113530075794, 0.8569339353867657, 0.8413298930280062, 0.8277603612320593, 0.8447839595764124, 0.828861581465355, 0.8421061119174326, 0.8316556469386658, 0.8217754125678652, 0.7937749825296996, 0.7636115680266624, 0.8181424501424501, 0.8032702252324894, 0.8339907541794334, 0.8224243401601893, 0.8337011234747085, 0.8254134279417297, 0.845977207977208, 0.8131213245175509, 0.8294037520829974, 0.8157383217760577, 0.8047652529161964, 0.8413416115680267, 0.8155741547062302, 0.8082849002849002, 0.8444005805515239, 0.7919376444659463, 0.8501552437778853, 0.7925077675643714, 0.8442291028328764, 0.8177190775681341, 0.7573595656614525, 0.7893066709670483, 0.7862699564586357, 0.8651101435252379, 0.7748609364081063, 0.8377571359458151, 0.8180081707251519, 0.8583311293877331, 0.8063031769069505, 0.856348653442993, 0.7908220179540935, 0.8534898672257162, 0.8157510079019513, 0.8124646562382412, 0.864213728968446, 0.8352431328280385, 0.8155597484276729, 0.8374389077030587, 0.7919947320324678]
Ori_Accuracy= [0.7518131868131868, 0.7652747252747252, 0.8054578754578754, 0.7558974358974359, 0.7554395604395605, 0.7728937728937729, 0.8072710622710624, 0.75, 0.7806043956043955, 0.7405677655677655, 0.7711904761904762, 0.7443406593406594, 0.7575274725274725, 0.7768681318681319, 0.7843406593406593, 0.7595787545787546, 0.7443223443223443, 0.7882051282051282, 0.7804578754578755, 0.8071428571428572, 0.7747435897435898, 0.7421428571428572, 0.7787362637362638, 0.79003663003663, 0.7823992673992674, 0.7385347985347985, 0.7348534798534798, 0.8034065934065934, 0.7709157509157508, 0.7500183150183151, 0.7421978021978022, 0.790091575091575, 0.8014835164835166, 0.7863003663003664, 0.7234249084249085, 0.7880769230769231, 0.732967032967033, 0.7595238095238096, 0.7804578754578755, 0.74996336996337, 0.7443040293040293, 0.7785897435897435, 0.7443589743589742, 0.728956043956044, 0.7403663003663004, 0.7288461538461538, 0.7537545787545789, 0.7500915750915752, 0.7843040293040293, 0.7615384615384615, 0.7174175824175824, 0.772838827838828, 0.7805677655677655, 0.7079120879120879, 0.7021062271062272, 0.7785714285714287, 0.7271428571428571, 0.7612820512820513, 0.751886446886447, 0.7727289377289377, 0.7405311355311356, 0.7728754578754579, 0.7423260073260074, 0.7728937728937729, 0.7672893772893774, 0.7881135531135531, 0.7308424908424909, 0.7615384615384615, 0.7938461538461539, 0.7443040293040293, 0.7461538461538462, 0.7348168498168498, 0.7231501831501832, 0.7861172161172162, 0.7727838827838829, 0.7384798534798535, 0.711886446886447, 0.7214285714285713, 0.7844322344322343, 0.769120879120879, 0.7957509157509157, 0.7692307692307693, 0.7862820512820512, 0.7271794871794872, 0.7728205128205128, 0.7251648351648352, 0.7519047619047619, 0.7518131868131868, 0.793992673992674, 0.7613919413919414, 0.7518498168498169, 0.7328937728937729, 0.7347985347985347, 0.7863553113553113, 0.7821611721611722, 0.7500915750915752, 0.769120879120879, 0.7939194139194139, 0.811025641025641, 0.7652930402930402, 0.7538095238095238, 0.7405677655677657, 0.736849816849817, 0.7519230769230769, 0.7672527472527472, 0.7613369963369963, 0.7767399267399268, 0.7996153846153847, 0.7481868131868132, 0.8187362637362637, 0.7632234432234432, 0.7729120879120879, 0.793956043956044, 0.7363186813186813, 0.7557692307692307, 0.728919413919414, 0.7977655677655677, 0.7671062271062271, 0.7901465201465202, 0.7157142857142857, 0.7767582417582417, 0.7632600732600732, 0.7537362637362637, 0.7612087912087913, 0.7176739926739927, 0.7098351648351648, 0.7804761904761904, 0.7101831501831503, 0.7882234432234432, 0.7805311355311356, 0.7593956043956043, 0.7442857142857143, 0.7652014652014651, 0.7518864468864469, 0.7613736263736264, 0.7634615384615385, 0.7995604395604395, 0.7843589743589744, 0.753901098901099, 0.7328205128205129, 0.7784798534798535, 0.7576373626373626, 0.7707509157509158, 0.7423809523809524, 0.7024542124542125, 0.7481684981684982, 0.7880586080586081, 0.7767399267399268, 0.7501465201465202, 0.7116666666666667, 0.7805494505494506, 0.8263003663003662, 0.772838827838828, 0.7577106227106227, 0.7728388278388277, 0.7652930402930402, 0.7805494505494505, 0.767051282051282, 0.7346336996336996, 0.7347619047619047, 0.7119230769230769, 0.7557142857142857, 0.7387728937728938, 0.7804761904761903, 0.770970695970696, 0.770934065934066, 0.753901098901099, 0.7939743589743591, 0.7557875457875458, 0.7635347985347986, 0.7463003663003663, 0.7306043956043956, 0.7766666666666666, 0.7711355311355311, 0.7518864468864469, 0.7768131868131867, 0.727014652014652, 0.7996520146520147, 0.7158424908424909, 0.7632417582417581, 0.7385347985347985, 0.7175274725274725, 0.70996336996337, 0.7386263736263736, 0.7805311355311355, 0.7252564102564103, 0.7844139194139194, 0.7805311355311356, 0.7901465201465202, 0.7442673992673993, 0.7956410256410257, 0.7173076923076923, 0.8033699633699634, 0.7557509157509157, 0.747985347985348, 0.7938644688644689, 0.768974358974359, 0.7557326007326007, 0.7651282051282051, 0.7325274725274726]



sensitivity = ['sensitivity' for _ in range(num)]
specificity = ['specificity' for _ in range(num)]
Area = ['auc' for _ in range(num)]
Accuracy = ['accuracy' for _ in range(num)]

Graph_festures=["Graph features" for _ in range(num)]
Original_fetures=["Original features" for _ in range(num)]



slop_sensitivity_val = pd.DataFrame({'Value': Graph_sensitivity, 'Fetures': Graph_festures, 'S': sensitivity})
slop_specificity_val = pd.DataFrame({'Value': Graph_specificity, 'Fetures': Graph_festures, 'S': specificity})
slop_Area_val = pd.DataFrame({'Value': Graph_Area, 'Fetures': Graph_festures, 'S': Area})
slop_Accuracy_val = pd.DataFrame({'Value': Graph_Accuracy, 'Fetures': Graph_festures, 'S': Accuracy})

density_sensitivity_val = pd.DataFrame({'Value': Ori_sensitivity, 'Fetures': Original_fetures, 'S': sensitivity})
density_specificity_val = pd.DataFrame({'Value': Ori_specificity, 'Fetures': Original_fetures, 'S': specificity})
density_Area_val = pd.DataFrame({'Value': Ori_Area, 'Fetures': Original_fetures, 'S': Area})
density_Accuracy_val = pd.DataFrame({'Value': Ori_Accuracy, 'Fetures': Original_fetures, 'S': Accuracy})




# slop_gmean_val,density_gmean_val,clustering_gmean_val, assortativity_gmean_val,
data = pd.concat([slop_sensitivity_val, slop_specificity_val,  slop_Area_val,slop_Accuracy_val,
                  density_sensitivity_val, density_specificity_val,  density_Area_val, density_Accuracy_val])
my_pal = {"Graph features": "r", "Original features": "y"}
sns.boxplot(x="S", y="Value", data=data, hue="Fetures",  width=0.7,palette=my_pal, linewidth=0.5)
plt.ylim(0.4,1)
plt.legend(bbox_to_anchor=(-0.036,1.05,1.069,0), loc=10,
                mode="expand", borderaxespad=1, ncol=4)
plt.plot([0.5,0.5], [0,1],c='black')
plt.plot([1.5,1.5], [0,1],c='black')
plt.plot([2.5,2.5], [0,1],c='black')
plt.plot([3.5,3.5], [0,1],c='black')
plt.xlabel(s="")
plt.ylabel("Performance")
plt.savefig('Fig07.eps', dpi=400, format='eps')
plt.show()