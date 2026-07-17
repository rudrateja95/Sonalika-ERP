async function deleteCart(id) {

    const confirmDelete = await Swal.fire({
        title: "Delete Item?",
        text: "Do you want to remove this item from the cart?",
        icon: "warning",
        showCancelButton: true,
        confirmButtonText: "Yes, Delete",
        cancelButtonText: "Cancel"
    });

    if (!confirmDelete.isConfirmed) {
        return;
    }

    const res = await fetch(`/api/cart/${id}`, {
        method: "DELETE"
    });

    const result = await res.json();

    if (result.status === "success") {

        Swal.fire({
            icon: "success",
            title: "Deleted",
            text: result.message,
            timer: 1500,
            showConfirmButton: false
        }).then(() => {

            loadCart();     // Reload table

        });

    } else {

        Swal.fire({
            icon: "error",
            title: "Error",
            text: result.message
        });

    }

}