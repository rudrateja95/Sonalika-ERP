function saveCart() {

    var payload = {
        items: getOrderData()
    };

    fetch("/api/cart", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(payload)
    })
    .then(function (res) {
        return res.json();
    })
    .then(function (result) {

        if (result.status === "success") {

            Swal.fire({
                icon: "success",
                title: "Success!",
                text: result.message,
                timer: 2000,
                showConfirmButton: false
            }).then(function () {

                // Close Modal
                $("#selectModal").modal("hide");

                // Reset Form
                document.getElementById("selectForm").reset();

                // Update Cart Count
                loadCartCount();

            });

        } else {

            Swal.fire({
                icon: "error",
                title: "Error!",
                text: result.message
            });

        }

    })
    .catch(function (error) {

        console.error(error);

        Swal.fire({
            icon: "error",
            title: "Network Error",
            text: "Unable to connect to the server."
        });

    });

}