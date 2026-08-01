function saveCart() {

    var payload = {
        items: getOrderData()
    };

    $.ajax({
        url: "/api/cart",
        type: "POST",
        contentType: "application/json",
        data: JSON.stringify(payload),

        success: function (result) {

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

        },

        error: function (xhr, status, error) {

            console.log(xhr.responseText);
            console.log(status);
            console.log(error);

            Swal.fire({
                icon: "error",
                title: "Network Error",
                text: "Unable to connect to the server."
            });

        }

    });

}