document.addEventListener("DOMContentLoaded", function () {
    window.addEventListener('scroll', function () {
        if (window.scrollY > 700) {
            document.getElementById('navbar_top').classList.add('fixed-top');
            // add padding top to show content behind navbar
            navbar_height = document.querySelector('.navbar').offsetHeight;
            document.body.style.paddingTop = navbar_height + 'px';
        } else {
            document.getElementById('navbar_top').classList.remove('fixed-top');
            // remove padding top from body
            document.body.style.paddingTop = '0';
        }
    });
});

function fill_sample_smile() {
    document.getElementById("smilesTextarea").value = 'CC(CCC(=O)N)CN\nC1=CC=CC=C1';
};

$(document).ready(function() {
    $('.dropdown-item').click(function() {
      var targetId = $(this).data('target');
      $('#' + targetId).button('toggle');
    });
  });