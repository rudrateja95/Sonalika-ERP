function saveCart() {

    var payload = {
        items: getOrderData()
    };

    var xhr = new XMLHttpRequest();

    xhr.open("POST", "/api/cart", true);
    xhr.setRequestHeader("Content-Type", "application/json");

    xhr.onreadystatechange = function () {

        if (xhr.readyState === 4) {

            if (xhr.status === 200) {

                var result;

                try {
                    result = JSON.parse(xhr.responseText);
                } catch (e) {

                    Swal.fire({
                        icon: "error",
                        title: "Error",
                        text: "Invalid server response."
                    });

                    return;
                }

                if (result.status === "success") {

                    Swal.fire({
                        icon: "success",
                        title: "Success!",
                        text: result.message,
                        timer: 2000,
                        showConfirmButton: false
                    }).then(function () {

                        $("#selectModal").modal("hide");

                        document.getElementById("selectForm").reset();

                        loadCartCount();

                    });

                } else {

                    Swal.fire({
                        icon: "error",
                        title: "Error!",
                        text: result.message
                    });

                }

            } else {

                Swal.fire({
                    icon: "error",
                    title: "Network Error",
                    text: "Unable to connect to the server."
                });

            }

        }

    };

    xhr.send(JSON.stringify(payload));

}