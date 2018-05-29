//function for getting randum No incl min and incl max 
function getRandomInt(min, max) {
    return Math.floor(Math.random() * (max - min + 1)) + min;
}

//function for replacing the target
function resetTarget(){
    $( "#l1" ).hide();
    nt= getRandomInt(0,h-64); 
    nl=  getRandomInt(0,w-64);
    $("#l1").css({top: nt, left: nl});
    setTimeout(function(){
        $( "#l1" ).show();
    }, 300);
}

//Initiate -> Get Window size
wstr = $( "#main" ).css( "width" );
hstr = $( "#main" ).css( "height" );
w=wstr.substring(0, wstr.length-2);
h=hstr.substring(0, hstr.length-2);
clicks = 0;

//Ad callback
$( "#target" ).click(function(){
    clicks ++;
    if (clicks >= 50){
        alert('done');
    }else{
        resetTarget()
    }
}
);
resetTarget();
