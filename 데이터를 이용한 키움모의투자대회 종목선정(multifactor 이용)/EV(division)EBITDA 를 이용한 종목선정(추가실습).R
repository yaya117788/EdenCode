'''
EV/EBITDA

-기업의 총 가치가 영업이익의 몇배인지 나타내주는 지표이기도 하다.

-값이 작을수록 저평가되어 있다고 볼 수 있다.

-기업 매매가 산정 시 가장 많이 사용됨

-값이 2라면 투자원금으로 시가총액을 회수하고 부채까지 상환하는데 걸리는 기간이 2년이라는 의미

-PER와 주가현금흐름비율(PCR)을 보완하므로 널리 사용됨

-EV/EBITDA가 시장평균을 하회한다면 이 기업의 내재가치는 저평가 되었다고 볼 수 있다.

-PER는 감가상각 등 실제 현금이 나가지 않는 부분을 장부에 반영하여 산출된 수치라고 볼 수 있어 실제 투자원금

회수와는 무관하여 EV/EBITDA 라는 세련된 평가기법을 활용한다.

​

NOTE

1.PER 주가수익비율과 차이점은 순손실이 발생했을 경우  PER는 계산이 어렵지만 EV/EBITDA는 계산가능

2.과도한 투자나 부채 등으로 재무건전성이 악화된 경우 가려내기 어렵다는 단점도 있다.

3.하지만 그럼에도 불구하고 증권사 리포트가 기업분석 시 빠지지 않고 사용하는 지표이므로 매우 중요하다.
[출처] 미국주식 재무제표 영업이익/EBIT/EV/EBITDA|작성자 왓피
'''

library(stringr)
library(ggplot2)
library(PerformanceAnalytics)
library(dplyr)
library(tidyr)
KOR_fs = readRDS('data/KOR_fs.Rds')
KOR_ticker = read.csv('data/KOR_ticker.csv', row.names = 1,
                      stringsAsFactors = FALSE) 

KOR_ticker$'종목코드' =
  str_pad(KOR_ticker$'종목코드', 6, 'left', 0)

## 현재의 월이 1~4월일경우 아직 전년도의 재무제표가 발표되지 않은 상황이다. 그래서 
## 이때 보수적으로 전년도가 아닌 전전년도 회계데이터를 사용한다.
if ( lubridate::month(Sys.Date()) %in% c(1,2,3,4) ) {
  num_col = ncol(KOR_fs[[1]]) - 1
} else {
  num_col = ncol(KOR_fs[[1]]) 
}

fs_item
## 우리가 사용할 제무제표 항목들의 재무제표상 이름을 확인하기 위해 fs_item으로 불러와서 확인.
''' EV는 기업가치로 시가총액 + 차입금(장/단기차입금,유동성장기부채,사채) - 현금성자산(현금및현금성자산, 단기금융상품,단기투자자산)으로 구한다.'''
''' EBITDA는 영업이익(EBIT) + 감가상각비(D) + 무형자산상각비(A)로 구한다.'''

(KOR_fs$매출액 - KOR_fs$매출원가)[1,num_col]
KOR_fs$매출총이익[1,num_col]
'''매출총이익 = 매출액 - 매출원가 확인'''

EBITDA <- (KOR_fs$매출총이익 - KOR_fs$판매비와관리비 + KOR_fs$감가상각비 +KOR_fs$무형자산상각비)[num_col]
EV <- (evvalue$시가총액.원./100000000) + ((KOR_fs$단기차입금 + KOR_fs$장기차입금 + KOR_fs$사채 + KOR_fs$유동성장기부채) 
       - KOR_fs$현금및현금성자산)[num_col]

# KOR_ticker의 데이터의 시가총액 데이터를 쓰기위해 KOR_fs의 데이터와 같은 형식으로 dataframe을 형성 
evvalue <- KOR_ticker['시가총액.원.']
rownames(evvalue) = NULL
rownames(evvalue) = KOR_ticker[, '종목코드']
head(evvalue)

nrow(evvalue)
nrow(KOR_fs$매출액)
# row의 개수를 확인해 계산에 문제가 없는지 확인 
EBITDA
head(EV,20)

'''문제발생 : EV값을 구하는데 여러 재무제표항목을 끌어오다보니 NA데이터가 증가하였고 거의 모든 ROW 결과값이 NA로 나오게된다.
NA를 0으로 변경시켜 계산필요'''
#totalname <- c('매출총이익','판매비와관리비','감가상각비','무형자산상각비','단기차입금','장기차입금','사채','유동성장기부채','현금및현금성자산')
#class(totalname)
##for (nom in totalname){
#  print(nom)
#  print(KOR_fs$nom)
#  KOR_fs$i[num_col][is.na(KOR_fs$i[num_col])] <- 0
#  print(head(KOR_fs$i[num_col],10))
#}
## for문으로는 $이후 값을 자동으로 인식하지 못한다. > 해결 불가 

KOR_fs$매출총이익[num_col][is.na(KOR_fs$매출총이익[num_col])] <- 0
KOR_fs$판매비와관리비[num_col][is.na(KOR_fs$판매비와관리비[num_col])] <- 0
KOR_fs$감가상각비[num_col][is.na(KOR_fs$감가상각비[num_col])] <- 0
KOR_fs$무형자산상각비[num_col][is.na(KOR_fs$무형자산상각비[num_col])] <- 0
KOR_fs$단기차입금[num_col][is.na(KOR_fs$단기차입금[num_col])] <- 0
KOR_fs$장기차입금[num_col][is.na(KOR_fs$장기차입금[num_col])] <- 0
KOR_fs$사채[num_col][is.na(KOR_fs$사채[num_col])] <- 0
KOR_fs$현금및현금성자산[num_col][is.na(KOR_fs$현금및현금성자산[num_col])] <- 0
KOR_fs$유동성장기부채[num_col][is.na(KOR_fs$유동성장기부채[num_col])] <- 0






EV1 <- (evvalue$시가총액.원. / 100000000) + ((KOR_fs$단기차입금 + KOR_fs$장기차입금 + KOR_fs$사채 + KOR_fs$유동성장기부채) 
                           - KOR_fs$현금및현금성자산)[num_col]
# ticker 데이터를 통해가져온 데이터는 원의 단위로 되어있고 재무제표 데이터는 억원 단위로 되어있어 조정
EBITDA <- (KOR_fs$매출총이익 - KOR_fs$판매비와관리비 + KOR_fs$감가상각비 +KOR_fs$무형자산상각비)[num_col]

head(EV1,20)
head(EBITDA)

evebitda <- EV1/EBITDA
head(evebitda)
# 각 회사들의 EV/ EBITA 비율을 구해놓았다.

'''재무제표 상의 NA 데이터로 인해 몇가지 지표들이 빠져 정확한 값이라고 판별할 수 없다.
그러나 여러가지 지표등을 이용해 종목선정할때 참고자료로 이용할 수 있을 것 같다. 예를들어 마법공식, 멀티팩터로 종목을 선정 후
그 중 이 비율을 이용해 추릴 수 있을 것이다.'''
