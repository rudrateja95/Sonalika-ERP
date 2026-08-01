function addOrder() {

    var container = document.getElementById("orderContainer");
    var clone = container.firstElementChild.cloneNode(true);

    // Reset all select boxes
    var selects = clone.getElementsByTagName("select");
    for (var i = 0; i < selects.length; i++) {
        selects[i].selectedIndex = 0;
    }

    // Reset all textareas
    var textareas = clone.getElementsByTagName("textarea");
    for (var j = 0; j < textareas.length; j++) {
        textareas[j].value = "";
    }

    // Show remove button
    var removeBtn = clone.querySelector(".remove-order");
    if (removeBtn) {
        removeBtn.style.display = "block";
    }

    container.appendChild(clone);
}

function removeOrder(btn) {

    var orderItem = btn;

    while (orderItem && !hasClass(orderItem, "order-item")) {
        orderItem = orderItem.parentNode;
    }

    if (orderItem && orderItem.parentNode) {
        orderItem.parentNode.removeChild(orderItem);
    }
}

// Helper function for old browsers
function hasClass(element, className) {
    return (" " + element.className + " ").indexOf(" " + className + " ") > -1;
}
