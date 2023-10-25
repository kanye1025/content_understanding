import collections
import csv
import pickle

from unique_tag import UniqueTag


s1 = """
   <div class="post-content" data-v-28d77a7a><p>Crypto lending platform Celsius has reportedly filed for Chapter 11 bankruptcy, having notified individual U.S. state regulators on Wednesday, July 13.</p><p>The news was <a href="https://www.cnbc.com/2022/07/13/embattled-crypto-lender-celsius-informs-state-regulators-that-its-filing-for-bankruptcy-imminently-source-says-.html" target="_blank" rel="noopener nofollow">reported</a> by CNBC and referred to an unnamed source, who asked not to be named as the proceedings were private. </p><p>The source noted that Celsius plans to file the paperwork &#8220;imminently.&#8221; </p><p>The news comes just days after the lending platform replaced its previously hired law firm with Kirkland &amp; Ellis LLP, the same firm that assisted Voyager Digital <a href="https://cointelegraph.com/news/voyager-digital-files-for-chapter-11-bankruptcy-proposes-recovery-plan">with its bankruptcy filing</a> in the Southern District Court of New York last week. </p><p>It follows the news the platform had officially paid off all of its DeFi debts to Maker, Compound, and Aave this week, reducing its debt from $820 million to <a href="https://zapper.fi/account/0x8aceab8167c80cb8b3de7fa6228b889bb1130ee8?tab=dashboard" target="_blank" rel="noopener nofollow">zero</a> in a matter of weeks. </p><p>On Tuesday, Vermont&#8217;s Department of Financial Regulation (DFR) <a href="https://dfr.vermont.gov/consumer-alert/dfr-encourages-celsius-network-investors-proceed-caution" target="_blank" rel="noopener nofollow">issued</a> a warning against the troubled crypto lending firm, reminding consumers that the firm is not licensed to offer its services in the state. </p><p>The DFR also stated it believed the company was &#8220;deeply insolvent&#8221; and doesn&#8217;t possess &#8220;assets and liquidity&#8221; to fulfill its obligations toward the customers, and accused them of mismanaging customer funds by allocating them towards risky investments. </p><p>Vermont has become the sixth state in America to open an investigation into Celsius&#8217;s crypto interest rate accounts, joining the likes of Alabama, Kentucky, New Jersey, Texas and Washington.</p><p>Rumors of Celsius&#8217; insolvency began circulating last month after the crypto lender was forced to halt withdrawals due to &#8220;extreme market conditions&#8221; on June 13. </p><template data-name="subscription_form" data-type="law_decoded"></template></div>
   """
s2 = """
   <div class="ArticleBody-articleBody" id="RegularArticle-ArticleBody-5" data-module="ArticleBody"><span hidden="" aria-hidden="true" class="ArticleBody-extraData"><span hidden="" aria-hidden="true" class="ArticleBody-extraData"><span hidden="" aria-hidden="true" class="xyz-data">Tesla has another bull in its corner. Truist Securities on Wednesday initiated coverage of the electric vehicle maker with a buy rating and price target of $1,000. That represents a more than 40% upside from where shares closed at $711.12 during Wednesday's session. "We believe the company's best days, in terms of volume production, product innovation, and, especially AI innovations, are still down the road," wrote analyst William Stein in the July 13 note. Truist expects Tesla to continue to outperform expectations in vehicle deliveries in the coming years and reach 10 million units per year by 2030, according to the note. "Our view is based on our expectation for robust growth to continue as TSLA's culture has encouraged three sustainable growth drivers: vertical integration and rapid innovation, consistent introduction of new platforms, and a growing number of more capable &amp; efficient factories," said Stein. The firm sees significant upside in Tesla's AI innovations, which could turbocharge growth opportunities going forward, according to the note. These include advanced driver assistance systems, autonomous driving, AI computing services and AI robotics. "We estimate these high-growth, high-profit margin potential AI innovations are responsible for 43% of the equity's value," Stein said. Truist sees Tesla's profitability taking a step back in the second quarter of 2022 due to rising input costs and lower volumes from the shutdown of its Shanghai factory, as well as increased factory costs in Shanghai and Texas and bitcoin deflation. Still, the firm expects Tesla to rebound and reach peak profitability in the fourth quarter of 2023. The company may be able to reach that goal sooner, in the third quarter of 2022, given a 26% contribution margin, according to the note. Of course, there are risks to the firm's base case, including shifting consumer spending on luxury and electric vehicles, the potential for further Covid-related shutdowns, supply costs and constraints and increased competition. Truist is also considering catalysts such as upcoming delivery and quarterly results, daily tweets from CEO Elon Musk, the company's upcoming AI day on Sept. 30 and more.</span></span></span><span class="HighlightShare-hidden" style="top:0;left:0"></span></div>
   """

# 初始化
docs = []  # 查询出当前库中的所有文章，格式：[(sim_id_1, hash_value_1), (sim_id_2, hash_value_2)]

uni_tag = UniqueTag(docs)

# # 测试函数
# s1_simhash = Simhash(s1)
# s2_simhash = Simhash(s2)
# uni_tag.add_to_index('1', s1_simhash)
# uni_tag.add_to_index('2', s2_simhash)
# print(uni_tag.get_doc_similar_id(s1))

# 模拟服务流程
if uni_tag.index.bucket_size() == 0:
    # todo 初始化index
    pass
message = []
for doc_id, doc in message:
    sim_id = uni_tag.get_doc_similar_id(doc_id, doc)

csv.field_size_limit(2000000)
with open("/Users/tongdechao/Downloads/short_article_sample_20220719.csv") as f:
    reader = csv.reader(f)
    article_dict = collections.defaultdict(list)
    count = 0
    for row in reader:
        # print(str(count) + "\n")
        count += 1
        if row[0] == "f_id": continue
        doc = row[1]
        doc_id = row[0]
        sim_id = uni_tag.get_doc_similar_id(doc_id, doc)
        article_dict[sim_id].append(uni_tag.txt_clean(doc))

out_f = open("simhash_index_file.pickle", 'wb')
pickle.dump(uni_tag.index, out_f)
out_f.close()
in_f = open("simhash_index_file.pickle", 'rb')
index = pickle.load(in_f)
pass