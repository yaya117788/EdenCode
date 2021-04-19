//하나만 먼저 개발해보고 나머지에 적용
//첫번째 버튼
//ep(0)으로 첫번째 버튼을 찾아주는것 [0]은 안된다. 참고할것
for (let i = 0; i < $('.tab-button').length; i ++){
//    var이 아닌 let으로 해서 돌려야 잘 돌아감 (for문 안에서만 변수가 돌아감) var로 할시 다시 돌아갈때 이미 변수가 3이 되어있기 때문에 안된다.
    $('.tab-button').eq(i).click(function(){
    $('.tab-content').removeClass('show');
    $('.tab-button').removeClass('active');
    $('.tab-button').eq(i).addClass('active');
    $('.tab-content').eq(i).addClass('show');
});
}
