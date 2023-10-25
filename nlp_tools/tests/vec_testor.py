import os
import sys
sys.path.append(os.getcwd())
import nltk
nltk.download('punkt')

from frameworks.content_understanding.nlp_tools.word.word_vec import get_word_vec,get_synonyms
from frameworks.content_understanding.nlp_tools.text.text_vec import get_text_vec

word = 'btc'


word_vec = get_word_vec(word,lang='en')
print(word_vec.shape)
print(get_synonyms(word,k = 10,lang='en'))


text = 'Sienna Network Launches SiennaLand, a DeFi Lending Platform】SiennaLend, the latest market offering of Sienna Network, contributes to the maximum utilization of privacy-oriented tokens by members of the crypto community.What Is SiennaLend?The major limitation of the existing DeFi applications is the inability to adequately protect users’ privacy. Their sensitive information can be easily accessed by any person with minimal technical knowledge. The promotion of user privacy constitutes the major priority for the large-scale integration of DeFi innovations in the mainstream services. Sienna Network has developed a unique technological solution SiennaLend that allows achieving the desired level of privacy and permissionless in regards to Bitcoin and Monero lending. Variable interest and over-collateralized models are widely used for providing the required liquidity and protecting users’ privacy to the maximum degree.After providing the required liquidity to the lending market, users are able to start earning interest almost immediately. SL LP tokens are used for representing liquidity provided to users in regards to specific lending pools. For instance, if the liquidity is provided for the sATOM lending pool, they may receive SL-sATOM as a representation of their share. All users also have an opportunity to provide a specific percentage of their crypto assets as collateral. In particular, they may be able to borrow up to 80% of their collateral USD value. Band Protocol is used as an oracle provider for securely managing data and ensuring the proper integration with other services.Sienna Network’s Expansion PlansSienna Network demonstrates the effective utilization of the attracted $11.2 million investments for developing its initiative of integrating privacy-based tokens into DeFi operations. Sienna Network is also effective in addressing the problem of front-running present in Ethereum when some users may receive advantage by offering higher transaction fees. Sienna Network’s algorithms effectively prevent such risks by offering more sustainable solutions that will not allow obtaining such advantage at the expense of other users. At the moment, its major competitors include Uniswap and PancakeSwap, but Sienna Network claims to offer broader functionality to its users.The key benefits of using SiennaLend for members of the crypto community refer to the possibility of economizing transaction fees, enjoying the maximum variety of lending services, and maintaining their privacy. The loans against collateral may also be effective for dealing with market volatility in a more effective manner. As cryptocurrency-based loans tend to become one of the key components of the DeFi segment, SiennaLend can enjoy additional benefits in the long term as compared with alternative projects. Being a part of the Cosmos ecosystem, Sienna Network allows reaching the maximum level of interconnectedness with other blockchain services.Investing in Sienna Network: Pros and ConsIn terms of its market capitalization, Sienna Network remains one of the most volatile crypto projects within the past year. Its wSIENNA token has reached the historical maximum of about $55 at the end of October, 2021. However, the project appeared to be highly sensitive to the negative external factors, causing its rapid decline to the levels close to the initial price level prior to the bullish cycle. Technical analysis may allow identifying the major support and resistance levels that may determine the token’s price dynamics in the following months.Figure 1. wSIENNA/USD Chart (1-Year); Data Source – CoinMarketCapThe token has the major support level at the price of $4 that corresponds to the local bottom within the past year, and it is still significant for preventing the complete capitulation of wSIENNA’s holders. The recent innovations offered by the company may be significant for preventing the further decline of its capitalization in the market. However, most investors remain highly concerned with the general situation in the crypto market as well as the possibility to earn high market returns under such conditions.There are several resistance levels that will largely determine the long-term potential of wSIENNA in the future. The first one is the resistance level at the price of $15 that may justify the reversal of the current negative price trend. The second one is the important level of $31 that may indicate the high likelihood of the token’s rapid appreciation in the following weeks. The third one is the price level of $50 that will indicate the possibility of exceeding historical maximums. Overall, the investments in the project may be reasonable after successfully overcoming at least the first major resistance.'
text_vec = get_text_vec(text,'en')
print(text_vec.shape)





word = '比特币'


word_vec = get_word_vec(word,lang='zh')
print(word_vec.shape)
print(get_synonyms(word,k = 10,lang='zh'))


text = '【火币要闻】1. 巴拿马国民议会正式批准加密市场监管法案；2. 知名真人秀节目《美国偶像》提交NFT相关商标申请；3. 以太坊收入同比增长46%至24亿美元；4. 资产管理公司WisdomTree表示，一季度管理加密资产增长23%至3.24亿美元；5. 国际税务联盟列出 NFT 市场欺诈的“危险信号”。'
text_vec = get_text_vec(text,'zh')

print(text_vec.shape)

text = 'При этом снять в иностранной валюте можно только те деньги, которые поступили на счет или вклад до 9 марта 2022 года. Остальные средства банк выдаст только в рублях по рыночному курсу на день снятия. "Выдача иностранной валюты осуществляется в долларах США или евро, независимо от валюты вклада или счета", - добавили в Банке России.'
text_vec = get_text_vec(text,'ru')
print(text_vec.shape)