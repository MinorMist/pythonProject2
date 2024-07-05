function fill_sample_smile() {
    // Sample smiles array
    let arr = ["CC(=O)NCCC1=CNc2c1cc(OC)cc2", "CCc(c1)ccc2[n+]1ccc3c2[nH]c4c3cccc4", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "CC1=CN=C(C(=C1OC)C)CS(=O)C2=NC3=C(N2)C=CC(=C3)OC", "CCC[C@@H](O)CC/C=C/C=C/C#CC#C/C=C/CO"];
    document.getElementById("smilesTextarea").value = arr[(Math.floor(Math.random() * arr.length))];
};

window.smoothScroll = function (target) {
    var scrollContainer = target;
    do { //find scroll container
        scrollContainer = scrollContainer.parentNode;
        if (!scrollContainer) return;
        scrollContainer.scrollTop += 1;
    } while (scrollContainer.scrollTop == 0);

    var targetY = 0;
    do { //find the top of target relatively to the container
        if (target == scrollContainer) break;
        targetY += target.offsetTop;
    } while (target = target.offsetParent);

    scroll = function (c, a, b, i) {
        i++; if (i > 30) return;
        c.scrollTop = a + (b - a) / 30 * i;
        setTimeout(function () { scroll(c, a, b, i); }, 7);
    }
    // start scrolling
    scroll(scrollContainer, scrollContainer.scrollTop, targetY, 0);
}

function copyToClipboard(element) {
    var $temp = $("<input>");
    $("body").append($temp);
    $temp.val($(element).text()).select();
    document.execCommand("copy");
    $temp.remove();
    var tooltip = document.getElementById("myTooltip1");
    tooltip.innerHTML = "Copied to clipboard ";
};