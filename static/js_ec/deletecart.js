function deleteCart(id) {

    if (!confirm("Do you want to remove this item from the cart?")) {
        return;
    }

    var xhr = new XMLHttpRequest();

    xhr.open("DELETE", "/api/cart/" + id, true);

    xhr.onreadystatechange = function () {

        if (xhr.readyState === 4) {

            if (xhr.status === 200) {

                var result;

                try {
                    result = JSON.parse(xhr.responseText);
                } catch (e) {
                    alert("Invalid server response.");
                    return;
                }

                if (result.status === "success") {

                    alert(result.message);

                    loadCart();

                } else {

                    alert(result.message);

                }

            } else {

                alert("Failed to delete item.");

            }

        }

    };

    xhr.send(null);

}