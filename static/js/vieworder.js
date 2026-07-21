async function viewOrder(orderNo) {

    const res = await fetch(`/api/order/${orderNo}`);

    const items = await res.json();

    let html = "";

    items.forEach(item => {

        html += `
        <tr>

            <td>

                <img src="data:image/jpeg;base64,${item.image}"
                     width="70">

            </td>

            <td>${item.style_no}</td>

            <td>${item.qty}</td>

            <td>${item.gold_color}</td>

            <td>${item.gold_purity}</td>

            <td>${item.diamond_color}</td>

            <td>${item.diamond_clarity}</td>

            <td>${item.remarks}</td>

            <td>${item.net_wt}</td>

            <td>${item.gross_wt}</td>

            <td>

                ${
                    item.gem_chart
                    ? `<img src="data:image/jpeg;base64,${item.gem_chart}" width="70">`
                    : "-"
                }

            </td>

        </tr>
        `;

    });

    document.getElementById("orderDetails").innerHTML = html;

    new bootstrap.Modal(document.getElementById("orderModal")).show();

}