function loadCart() {

    // Show Loader
    document.getElementById("cartBody").innerHTML =
        '<tr>' +
        '<td colspan="11" class="text-center py-4">' +
        '<div class="spinner-border text-primary" role="status"></div>' +
        '<div class="mt-2">Loading cart...</div>' +
        '</td>' +
        '</tr>';

    fetch("/api/cart-list")
        .then(function (res) {
            return res.json();
        })
        .then(function (data) {

            var html = "";
            var totalQty = 0;

            for (var i = 0; i < data.length; i++) {

                var item = data[i];

                totalQty += Number(item.qty || 0);

                html +=
                    '<tr ' +
                    'data-id="' + item.id + '" ' +
                    'data-style-no="' + item.style_no + '" ' +
                    'data-qty="' + item.qty + '" ' +
                    'data-gold-color="' + item.gold_color + '" ' +
                    'data-gold-purity="' + item.gold_purity + '" ' +
                    'data-diamond-color="' + item.diamond_color + '" ' +
                    'data-diamond-clarity="' + item.diamond_clarity + '" ' +
                    'data-remarks="' + item.remarks + '">' +

                    '<td>' + (i + 1) + '</td>' +
                    '<td>' + item.style_no + '</td>' +
                    '<td>' + item.qty + '</td>' +
                    '<td>' + item.gold_color + '</td>' +
                    '<td>' + item.gold_purity + '</td>' +
                    '<td>' + item.diamond_color + '</td>' +
                    '<td>' + item.diamond_clarity + '</td>' +
                    '<td>' + item.remarks + '</td>' +

                    '<td>' +
                    '<button class="btn btn-sm btn-primary" onclick="editCart(' + item.id + ')">' +
                    '<i class="fas fa-edit"></i>' +
                    '</button>' +
                    '</td>' +

                    '<td>' +
                    '<button class="btn btn-sm btn-danger" onclick="deleteCart(' + item.id + ')">' +
                    '<i class="fas fa-trash"></i>' +
                    '</button>' +
                    '</td>' +

                    '</tr>';
            }

            if (html === "") {
                html =
                    '<tr>' +
                    '<td colspan="11" class="text-center">' +
                    'Cart is empty.' +
                    '</td>' +
                    '</tr>';
            }

            document.getElementById("cartBody").innerHTML = html;

        })
        .catch(function (err) {

            console.error(err);

            document.getElementById("cartBody").innerHTML =
                '<tr>' +
                '<td colspan="11" class="text-center text-danger">' +
                'Failed to load cart.' +
                '</td>' +
                '</tr>';

            Swal.fire({
                icon: "error",
                title: "Error",
                text: "Failed to load cart."
            });

        });

}

loadCart();