async function saveCart() {
    try {

        const payload = {
            client_code: "TEST001",
            items: getOrderData()
        };

        const res = await fetch("/api/cart", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(payload)
        });

        const result = await res.json();

        if (result.status === "success") {

            Swal.fire({
                icon: "success",
                title: "Success!",
                text: result.message,
                timer: 2000,
                showConfirmButton: false
            }).then(() => {

                // Close Modal
                $("#selectModal").modal("hide");

                // Reset Form
                document.getElementById("selectForm").reset();

                // Update Cart Quantity
                loadCartCount();

            });

        } else {

            Swal.fire({
                icon: "error",
                title: "Error!",
                text: result.message
            });

        }

    } catch (error) {

        console.error(error);

        Swal.fire({
            icon: "error",
            title: "Network Error",
            text: "Unable to connect to the server."
        });

    }
}