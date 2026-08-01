function deleteCart(id) {

    Swal.fire({
        title: "Delete Item?",
        text: "Do you want to remove this item from the cart?",
        icon: "warning",
        showCancelButton: true,
        confirmButtonText: "Yes, Delete",
        cancelButtonText: "Cancel"
    }).then(function (confirmDelete) {

        if (!confirmDelete.isConfirmed) {
            return;
        }

        fetch("/api/cart/" + id, {
            method: "DELETE"
        })
        .then(function (res) {
            return res.json();
        })
        .then(function (result) {

            if (result.status === "success") {

                Swal.fire({
                    icon: "success",
                    title: "Deleted",
                    text: result.message,
                    timer: 1500,
                    showConfirmButton: false
                }).then(function () {

                    loadCart();   // Reload table

                });

            } else {

                Swal.fire({
                    icon: "error",
                    title: "Error",
                    text: result.message
                });

            }

        })
        .catch(function (err) {

            console.error(err);

            Swal.fire({
                icon: "error",
                title: "Error",
                text: "Failed to delete item."
            });

        });

    });

}