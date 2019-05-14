import numpy as np


class Solution:
    def classification(self, text):
        """
        :type text: str
        :rtype: List[int]
        """
        if not text:
            return []
        data = text
        data = self.process(data)
        if len(data) == 1:
            return [0]
        # 获取词汇表，document frequency， 词频
        vocab, df, vb_count = self.vocabulary(data)
        # 去频率为1的词
        for line in data:
            for word in line:
                if vb_count[word] <= 1:
                    line.remove(word)
        # 重新获取
        vocab, df, vb_count = self.vocabulary(data)
        # 提取特征
        tfidf = self.get_tfidf(data, vocab, df)
        # 计算相似度
        sim = self.cosine_sim(tfidf)
        # 聚类
        res = hierarchical(np.array(tfidf), sim)
        return res

    def pipeline(self, line):
        import re
        # 去除非英文字符
        p1 = re.compile(r'-\{.*?(zh-hans|zh-cn):([^;]*?)(;.*?)?\}-')
        p2 = re.compile(r'[(][: @ . , ？！\s][)]')
        p3 = re.compile(r'[「『]')
        p4 = re.compile(r'[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）0-9 , : ; \-\ \[\ \]\ ]')
        line = p1.sub(r' ', line)
        line = p2.sub(r' ', line)
        line = p3.sub(r' ', line)
        line = p4.sub(r' ', line)
        # 分词
        line = [x.lower() for x in line.split(' ') if x]
        res = []
        stoplist = ["'d","'ll","'m","'re","'s","'t","'ve","ZT","ZZ","a","a's","able","about","above","abst","accordance","according","accordingly","across","act","actually","added","adj","adopted","affected","affecting","affects","after","afterwards","again","against","ah","ain't","all","allow","allows","almost","alone","along","already","also","although","always","am","among","amongst","an","and","announce","another","any","anybody","anyhow","anymore","anyone","anything","anyway","anyways","anywhere","apart","apparently","appear","appreciate","appropriate","approximately","are","area","areas","aren","aren't","arent","arise","around","as","aside","ask","asked","asking","asks","associated","at","auth","available","away","awfully","b","back","backed","backing","backs","be","became","because","become","becomes","becoming","been","before","beforehand","began","begin","beginning","beginnings","begins","behind","being","beings","believe","below","beside","besides","best","better","between","beyond","big","biol","both","brief","briefly","but","by","c","c'mon","c's","ca","came","can","can't","cannot","cant","case","cases","cause","causes","certain","certainly","changes","clear","clearly","co","com","come","comes","concerning","consequently","consider","considering","contain","containing","contains","corresponding","could","couldn't","couldnt","course","currently","d","date","definitely","describe","described","despite","did","didn't","differ","different","differently","discuss","do","does","doesn't","doing","don't","done","down","downed","downing","downs","downwards","due","during","e","each","early","ed","edu","effect","eg","eight","eighty","either","else","elsewhere","end","ended","ending","ends","enough","entirely","especially","et","et-al","etc","even","evenly","ever","every","everybody","everyone","everything","everywhere","ex","exactly","example","except","f","face","faces","fact","facts","far","felt","few","ff","fifth","find","finds","first","five","fix","followed","following","follows","for","former","formerly","forth","found","four","from","full","fully","further","furthered","furthering","furthermore","furthers","g","gave","general","generally","get","gets","getting","give","given","gives","giving","go","goes","going","gone","good","goods","got","gotten","great","greater","greatest","greetings","group","grouped","grouping","groups","h","had","hadn't","happens","hardly","has","hasn't","have","haven't","having","he","he's","hed","hello","help","hence","her","here","here's","hereafter","hereby","herein","heres","hereupon","hers","herself","hes","hi","hid","high","higher","highest","him","himself","his","hither","home","hopefully","how","howbeit","however","hundred","i","i'd","i'll","i'm","i've","id","ie","if","ignored","im","immediate","immediately","importance","important","in","inasmuch","inc","include","indeed","index","indicate","indicated","indicates","information","inner","insofar","instead","interest","interested","interesting","interests","into","invention","inward","is","isn't","it","it'd","it'll","it's","itd","its","itself","j","just","k","keep","keeps","kept","keys","kg","kind","km","knew","know","known","knows","l","large","largely","last","lately","later","latest","latter","latterly","least","less","lest","let","let's","lets","like","liked","likely","line","little","long","longer","longest","look","looking","looks","ltd","m","made","mainly","make","makes","making","man","many","may","maybe","me","mean","means","meantime","meanwhile","member","members","men","merely","mg","might","million","miss","ml","more","moreover","most","mostly","mr","mrs","much","mug","must","my","myself","n","n't","na","name","namely","nay","nd","near","nearly","necessarily","necessary","need","needed","needing","needs","neither","never","nevertheless","new","newer","newest","next","nine","ninety","no","nobody","non","none","nonetheless","noone","nor","normally","nos","not","noted","nothing","novel","now","nowhere","number","numbers","o","obtain","obtained","obviously","of","off","often","oh","ok","okay","old","older","oldest","omitted","in","on","once","one","ones","only","onto","open","opened","opening","opens","or","ord","order","ordered","ordering","orders","other","others","otherwise","ought","our","ours","ourselves","out","outside","over","overall","owing","own","p","page","pages","part","parted","particular","particularly","parting","parts","past","per","perhaps","place","placed","places","please","plus","point","pointed","pointing","points","poorly","possible","possibly","potentially","pp","predominantly","present","presented","presenting","presents","presumably","previously","primarily","probably","problem","problems","promptly","proud","provides","put","puts","q","que","quickly","quite","qv","r","ran","rather","rd","re","readily","really","reasonably","recent","recently","ref","refs","regarding","regardless","regards","related","relatively","research","respectively","resulted","resulting","results","right","room","rooms","run","s","said","same","saw","say","saying","says","sec","second","secondly","seconds","section","see","seeing","seem","seemed","seeming","seems","seen","sees","self","selves","sensible","sent","serious","seriously","seven","several","shall","she","she'll","shed","shes","should","shouldn't","show","showed","showing","shown","showns","shows","side","sides","significant","significantly","similar","similarly","since","six","slightly","small","smaller","smallest","so","some","somebody","somehow","someone","somethan","something","sometime","sometimes","somewhat","somewhere","soon","sorry","specifically","specified","specify","specifying","state","states","still","stop","strongly","sub","substantially","successfully","such","sufficiently","suggest","sup","sure","t","t's","take","taken","taking","tell","tends","th","than","thank","thanks","thanx","that","that'll","that's","that've","thats","the","their","theirs","them","themselves","then","thence","there","there'll","there's","there've","thereafter","thereby","thered","therefore","therein","thereof","therere","theres","thereto","thereupon","these","they","they'd","they'll","they're","they've","theyd","theyre","thing","things","think","thinks","third","this","thorough","thoroughly","those","thou","though","thoughh","thought","thoughts","thousand","three","throug","through","throughout","thru","thus","til","tip","to","today","together","too","took","toward","towards","tried","tries","truly","try","trying","ts","turn","turned","turning","turns","twice","two","u","un","under","unfortunately","unless","unlike","unlikely","until","unto","up","upon","ups","us","use","used","useful","usefully","usefulness","uses","using","usually","uucp","v","value","various","very","via","viz","vol","vols","vs","w","want","wanted","wanting","wants","was","wasn't","way","ways","we","we'd","we'll","we're","we've","wed","welcome","well","wells","went","were","weren't","what","what'll","what's","whatever","whats","when","whence","whenever","where","where's","whereafter","whereas","whereby","wherein","wheres","whereupon","wherever","whether","which","while","whim","whither","who","who'll","who's","whod","whoever","whole","whom","whomever","whos","whose","why","widely","will","willing","wish","with","within","without","won't","wonder","words","work","worked","working","works","world","would","wouldn't","www","x","y","year","years","yes","yet","you","you'd","you'll","you're","you've","youd","young","younger","youngest","your","youre","yours","yourself","yourselves","z","zero","zt","zz"]
        for word in line:
            # 去停用词
            if word in stoplist:
                continue
            # 对缩写进行补充
            if word == 'cmb':
                res += ['china', 'merchants', 'bank']
            res.append(word)
        return res

    def process(self,data):
        # 按句子切分
        data = list(filter(lambda m: m, data.split('\n')))
        res = []
        for line in data:
            res.append(self.pipeline(line))
        return res

    def vocabulary(self,sen_list):
        vb_count = {}
        vocab = {}
        df = {}
        idx = 0
        for line in sen_list:
            flag = {word: 0 for word in line}
            for word in line:
                if word not in vocab:
                    vocab[word] = idx
                    idx += 1
                    vb_count[word] = 1
                    df[word] = 1
                    flag[word] = 1
                else:
                    vb_count[word] += 1
                    if flag[word] == 0:
                        df[word] += 1
                        flag[word] = 1
        return vocab, df, vb_count

    def get_tfidf(self, sen_list, vocab, df_vocab):
        import math
        res = []
        for sen in sen_list:
            sen_tf_idf = [0.0 for _ in range(len(vocab))]
            for word in set(sen):
                tf = sen.count(word) / len(sen)
                # 对位置靠前的词语赋予高权重
                if sen.index(word) < len(sen)*0.15:
                    tf *= 5
                # 改进的tfidf，因为段落数较少，词汇表较小，在去除停用词的基础上，
                # 文档之间共有的词更能体现文档的相似性
                tfidf = tf * math.sqrt(df_vocab[word])
                sen_tf_idf[vocab[word]] = tfidf
            res.append(sen_tf_idf)
        return res

    def cosine_sim(self, matrix):
        import numpy as np
        mat = np.array(matrix)
        # avg = np.average(mat, 0)
        # mat = mat - avg
        fenzi = np.matmul(mat, mat.T)
        fanshu = np.linalg.norm(mat, axis=1, keepdims=True)
        fenmu = np.matmul(fanshu, fanshu.T)
        sim = fenzi / fenmu
        return sim


def find_max_sim(temp_res, dist):
    max_i = 0
    max_j = 0
    max_sim = -99999
    for i in range(len(temp_res)-1):
        for j in range(i+1, len(temp_res)):
            temp_dist = avg_similarity(temp_res[i], temp_res[j], dist)
            if max_sim < temp_dist:
                max_sim = temp_dist
                max_i = i
                max_j = j
    return max_i, max_j, max_sim


def avg_similarity(list1, list2, dist):
    # cosine similarity
    fenmu = len(list1)*len(list2)
    fenzi = 0
    for idx1 in list1:
        for idx2 in list2:
            fenzi += dist[idx1][idx2]
    return fenzi / fenmu


def hierarchical(data, dist, threshold=0.0):
    num_samples = data.shape[0]
    cluster_res = [[i] for i in range(num_samples)]
    while True:
        # 找到需要合并的簇
        max_i, max_j, max_sim = find_max_sim(cluster_res, dist)
        if max_sim <= threshold:
            break
        else:
            print('合并的簇是:', cluster_res[max_i], cluster_res[max_j], max_sim)
            cluster_res[max_i] += cluster_res[max_j]
            cluster_res.remove(cluster_res[max_j])
    res = [-1 for _ in range(num_samples)]
    for i, cluster in enumerate(cluster_res):
        for idx in cluster:
            res[idx] = i
    return res
    # return res, silhouette_coefficient(dist, cluster_res, res)


# 轮廓系数
def silhouette_coefficient(dist, clusters, res):
    dist = 1-dist
    nums_samples = len(res)
    coef = [0 for _ in range(nums_samples)]
    for i in range(nums_samples):
        idx_cluster = res[i]
        cluster_samples = clusters[idx_cluster]
        if len(cluster_samples) == 1:
            coef[i] = 0
            continue
        dis = 0.0
        max_dis = -99999
        for sample in cluster_samples:
            if sample == i:
                continue
            dis += dist[i][sample]
            if max_dis < dist[i][sample]:
                max_dis = dist[i][sample]
        a = dis/(len(cluster_samples)-1)

        if len(cluster_samples) == nums_samples:
            b = max_dis
        else:
            b = 99999
            for j, cluster in enumerate(clusters):
                if j == idx_cluster:
                    continue
                dis = 0.0
                for sample in cluster:
                    dis += dist[i][sample]
                dis = dis / len(cluster)
                if dis < b:
                    b = dis
        coef[i] = (b - a)/max(a, b)
    # print("----", sum(coef)/nums_samples)
    return sum(coef)/nums_samples
    # return coef


# ch分数
def ch_score(data, clusters):
    extra_disp, intra_disp = 0., 0.
    X = data
    n_labels = len(clusters)
    n_samples = len(data)
    if n_labels < 2 or n_labels > n_samples - 1:
        return float("inf")
    mean = np.mean(X, axis=0)
    for k in range(n_labels):
        cluster_k = X[clusters[k]]
        mean_k = np.mean(cluster_k, axis=0)
        extra_disp += len(cluster_k) * np.sum((mean_k - mean) ** 2)
        intra_disp += np.sum((cluster_k - mean_k) ** 2)

    return (1. if intra_disp == 0. else
            extra_disp * (n_samples - n_labels) /
            (intra_disp * (n_labels - 1.)))
