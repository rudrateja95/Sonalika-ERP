// Open Edit Modal
function editCart(id) {

    document.getElementById("edit_cart_id").value = id;

    fetch("/api/cart/" + id)
        .then(function (res) {
            return res.json();
        })
        .then(function (item) {

            document.querySelector('[data-role="edit_style_no"]').value = item.style_no || "";
            document.querySelector('[data-role="edit_qty"]').value = item.qty || "";
            document.querySelector('[data-role="edit_gold_color"]').value = item.gold_color || "";
            document.querySelector('[data-role="edit_gold_purity"]').value = item.gold_purity || "";
            document.querySelector('[data-role="edit_diamond_color"]').value = item.diamond_color || "";
            document.querySelector('[data-role="edit_diamond_clarity"]').value = item.diamond_clarity || "";
            document.querySelector('[data-role="edit_remarks"]').value = item.remarks || "";

            $("#editCartModal").modal("show");

        })
        .catch(function (err) {

            console.error(err);

            Swal.fire({
                icon: "error",
                title: "Error",
                text: "Unable to load cart item."
            });

        });

}